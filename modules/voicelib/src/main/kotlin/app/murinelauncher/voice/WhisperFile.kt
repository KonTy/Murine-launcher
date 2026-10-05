// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import java.io.EOFException
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Thrown when a file is not a usable whisper.cpp GGML model. */
class InvalidModelException(message: String) : IOException(message)

/** Hyperparameters from the header of a whisper.cpp GGML model file. */
data class WhisperHeader(
    val nVocab: Int,
    val nAudioCtx: Int,
    val nAudioState: Int,
    val nAudioHead: Int,
    val nAudioLayer: Int,
    val nTextCtx: Int,
    val nTextState: Int,
    val nTextHead: Int,
    val nTextLayer: Int,
    val nMels: Int,
    val ftype: Int,
) {
    /** English-only checkpoints have one token fewer than the multilingual ones. */
    val isMultilingual: Boolean get() = nVocab >= WhisperFile.MULTILINGUAL_VOCAB

    /** Quantization of the weights, without the quantization-version factor. */
    val weightType: Int get() = ftype % WhisperFile.QNT_VERSION_FACTOR

    val size: ModelSize get() = ModelSize.of(nAudioState, nAudioLayer)
}

/** OpenAI Whisper checkpoint sizes, recognised from the encoder width and depth. */
enum class ModelSize(val width: Int, val layers: Int) {
    TINY(384, 4), BASE(512, 6), SMALL(768, 12), MEDIUM(1024, 24), LARGE(1280, 32), UNKNOWN(0, 0);

    companion object {
        fun of(width: Int, layers: Int) =
            entries.firstOrNull { it.width == width && it.layers == layers } ?: UNKNOWN
    }
}

/**
 * Reads and validates the whisper.cpp GGML model format (magic `ggml`, as written by the
 * whisper.cpp conversion scripts and as published by FUTO for its fine-tuned models).
 *
 * [validate] walks the whole file the way the native loader does, without reading the weights,
 * so that a truncated download or a foreign file is rejected before it is installed and before
 * any native code allocates memory based on its contents.
 */
object WhisperFile {
    const val MAGIC = 0x67676d6c
    const val QNT_VERSION_FACTOR = 1000
    const val MULTILINGUAL_VOCAB = 51865
    const val HEADER_BYTES = 4 + 11 * 4

    private const val MIN_TENSORS = 10
    private const val MAX_VOCAB_ENTRY = 1024
    private const val MAX_TENSOR_NAME = 256
    private const val MAX_DIM = 1 shl 24

    // ggml_ftype values whisper.cpp can load
    private val SUPPORTED_FTYPES = setOf(0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14)

    // ggml_type id -> (block size, bytes per block)
    private val TYPE_SIZES = mapOf(
        0 to (1 to 4),      // F32
        1 to (1 to 2),      // F16
        2 to (32 to 18),    // Q4_0
        3 to (32 to 20),    // Q4_1
        6 to (32 to 22),    // Q5_0
        7 to (32 to 24),    // Q5_1
        8 to (32 to 34),    // Q8_0
        9 to (32 to 36),    // Q8_1
        10 to (256 to 84),  // Q2_K
        11 to (256 to 110), // Q3_K
        12 to (256 to 144), // Q4_K
        13 to (256 to 176), // Q5_K
        14 to (256 to 210), // Q6_K
        15 to (256 to 292), // Q8_K
        30 to (1 to 2),     // BF16
    )

    /** Parses and sanity-checks the header from the first [HEADER_BYTES] of a file. */
    @Throws(InvalidModelException::class)
    fun parseHeader(bytes: ByteArray, length: Int = bytes.size): WhisperHeader {
        if (length < HEADER_BYTES) throw InvalidModelException("File too short for a model header")
        val buf = ByteBuffer.wrap(bytes, 0, HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        if (buf.int != MAGIC) throw InvalidModelException("Not a whisper.cpp GGML model")
        val h = WhisperHeader(
            nVocab = buf.int, nAudioCtx = buf.int, nAudioState = buf.int, nAudioHead = buf.int,
            nAudioLayer = buf.int, nTextCtx = buf.int, nTextState = buf.int, nTextHead = buf.int,
            nTextLayer = buf.int, nMels = buf.int, ftype = buf.int,
        )
        checkHeader(h)
        return h
    }

    private fun checkHeader(h: WhisperHeader) {
        fun require(ok: Boolean, what: String) {
            if (!ok) throw InvalidModelException("Unsupported model: $what")
        }
        require(h.nVocab in 51_000..52_000, "vocabulary size ${h.nVocab}")
        require(h.nAudioCtx == 1500, "audio context ${h.nAudioCtx}")
        require(h.nTextCtx == 448, "text context ${h.nTextCtx}")
        require(h.nMels == 80 || h.nMels == 128, "mel bands ${h.nMels}")
        require(h.nAudioState == h.nTextState && h.nAudioState in 64..4096, "width ${h.nAudioState}")
        require(h.nAudioHead in 1..64 && h.nTextHead in 1..64, "attention heads")
        require(h.nAudioLayer in 1..64 && h.nTextLayer in 1..64, "layer count")
        require(h.ftype >= 0 && h.weightType in SUPPORTED_FTYPES, "weight type ${h.ftype}")
    }

    /** Reads only the header of [file]. */
    @Throws(IOException::class)
    fun readHeader(file: File): WhisperHeader = file.inputStream().use { input ->
        val bytes = ByteArray(HEADER_BYTES)
        parseHeader(bytes, input.readFully(bytes))
    }

    /**
     * Checks the whole file structure: header, mel filters, vocabulary and every tensor record,
     * which must end exactly at the end of the file.
     */
    @Throws(IOException::class)
    fun validate(file: File): WhisperHeader {
        val total = file.length()
        RandomAccessFile(file, "r").use { raf ->
            val reader = Reader(raf, total)
            val headerBytes = ByteArray(HEADER_BYTES)
            reader.read(headerBytes)
            val header = parseHeader(headerBytes)

            val nMel = reader.int()
            val nFft = reader.int()
            if (nMel != header.nMels || nFft !in 1..4096) throw InvalidModelException("Bad mel filters")
            reader.skip(nMel.toLong() * nFft * 4)

            val nVocab = reader.int()
            if (nVocab !in 0..60_000) throw InvalidModelException("Bad vocabulary")
            repeat(nVocab) {
                val len = reader.int()
                if (len !in 0..MAX_VOCAB_ENTRY) throw InvalidModelException("Bad vocabulary entry")
                reader.skip(len.toLong())
            }

            var tensors = 0
            while (reader.position < total) {
                val nDims = reader.int()
                val nameLength = reader.int()
                val type = reader.int()
                if (nDims !in 1..4) throw InvalidModelException("Bad tensor rank")
                if (nameLength !in 1..MAX_TENSOR_NAME) throw InvalidModelException("Bad tensor name")
                val (blockSize, blockBytes) = TYPE_SIZES[type]
                    ?: throw InvalidModelException("Unsupported tensor type $type")
                var elements = 1L
                repeat(nDims) {
                    val ne = reader.int()
                    if (ne !in 1..MAX_DIM) throw InvalidModelException("Bad tensor shape")
                    elements *= ne
                }
                if (elements % blockSize != 0L) throw InvalidModelException("Bad tensor shape")
                reader.skip(nameLength.toLong())
                reader.skip(elements / blockSize * blockBytes)
                tensors++
            }
            if (tensors < MIN_TENSORS) throw InvalidModelException("Model has no weights")
            return header
        }
    }

    private class Reader(private val raf: RandomAccessFile, private val total: Long) {
        private val scratch = ByteArray(4)
        // Small buffered window: the vocabulary is tens of thousands of tiny records
        private val buffer = ByteArray(64 * 1024)
        private var bufferStart = 0L
        private var bufferLength = 0
        var position = 0L
            private set

        fun read(dst: ByteArray) {
            var done = 0
            while (done < dst.size) {
                if (position < bufferStart || position >= bufferStart + bufferLength) fill()
                val offset = (position - bufferStart).toInt()
                val n = minOf(dst.size - done, bufferLength - offset)
                System.arraycopy(buffer, offset, dst, done, n)
                done += n
                position += n
            }
        }

        fun int(): Int {
            read(scratch)
            return (scratch[0].toInt() and 0xff) or
                ((scratch[1].toInt() and 0xff) shl 8) or
                ((scratch[2].toInt() and 0xff) shl 16) or
                ((scratch[3].toInt() and 0xff) shl 24)
        }

        fun skip(bytes: Long) {
            if (bytes < 0 || position + bytes > total) throw InvalidModelException("Model file is truncated")
            position += bytes
        }

        private fun fill() {
            if (position >= total) throw InvalidModelException("Model file is truncated")
            raf.seek(position)
            val n = raf.read(buffer, 0, buffer.size)
            if (n <= 0) throw EOFException("Model file is truncated")
            bufferStart = position
            bufferLength = n
        }
    }
}

/** Reads until [dst] is full or the stream ends; returns the number of bytes read. */
internal fun InputStream.readFully(dst: ByteArray): Int {
    var done = 0
    while (done < dst.size) {
        val n = read(dst, done, dst.size - done)
        if (n < 0) break
        done += n
    }
    return done
}

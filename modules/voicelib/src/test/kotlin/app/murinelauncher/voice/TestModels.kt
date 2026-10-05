// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Builds small, structurally valid whisper.cpp GGML files for tests. */
object TestModels {

    fun bytes(
        magic: Int = WhisperFile.MAGIC,
        nVocab: Int = 51864,
        width: Int = 384,
        layers: Int = 4,
        audioCtx: Int = 1500,
        ftype: Int = 2007,
        tensors: Int = 12,
        tensorType: Int = 8,
        seed: Int = 0,
    ): ByteArray {
        val out = ByteArrayOutputStream()
        fun int(v: Int) = out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array())

        int(magic)
        listOf(nVocab, audioCtx, width, 6, layers, 448, width, 6, layers, 80, ftype).forEach(::int)
        // Mel filters: n_mel, n_fft, floats
        int(80); int(2)
        repeat(80 * 2) { int(0) }
        // Vocabulary
        int(100)
        repeat(100) { int(1); out.write('a'.code + it % 26) }
        // Tensors: Q8_0 [32 x 2] = 2 blocks of 34 bytes
        repeat(tensors) { i ->
            val name = "tensor.$i".toByteArray()
            int(2); int(name.size); int(tensorType)
            int(32); int(2)
            out.write(name)
            val (block, size) = if (tensorType == 0) 1 to 4 else 32 to 34
            out.write(ByteArray(64 / block * size) { (it + i + seed).toByte() })
        }
        return out.toByteArray()
    }
}

// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build

/**
 * Microphone input through [AudioRecord], in memory only. Takes transient exclusive audio focus
 * while listening, so music pauses and comes back afterwards, and reports the input as interrupted
 * when focus is lost (a call) or the system silences it (another app took the microphone).
 */
class MicAudioSource(context: Context) : AudioSource {

    private val appContext = context.applicationContext
    private val audioManager = appContext.getSystemService(AudioManager::class.java)
    private val lock = Any()
    @Volatile private var record: AudioRecord? = null
    private var focusRequest: AudioFocusRequest? = null
    @Volatile private var focusLost = false
    @Volatile private var stopped = false

    override fun start(): Boolean {
        if (appContext.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return false
        }
        synchronized(lock) {
            if (stopped || record != null) return false
            val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
            if (minBuffer <= 0) return false
            val r = try {
                @Suppress("MissingPermission")
                AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE, CHANNEL, ENCODING,
                    maxOf(minBuffer * 2, BUFFER_BYTES)
                )
            } catch (_: Exception) {
                return false
            }
            if (r.state != AudioRecord.STATE_INITIALIZED) {
                r.release()
                return false
            }
            record = r
            requestFocus()
            try {
                r.startRecording()
            } catch (_: Exception) {
                return false
            }
            return r.recordingState == AudioRecord.RECORDSTATE_RECORDING
        }
    }

    override fun read(buffer: ShortArray, offset: Int, length: Int): Int {
        // Not under the lock: read blocks until audio arrives or stop() is called
        val r = record ?: return -1
        if (stopped) return -1
        return try {
            r.read(buffer, offset, length)
        } catch (_: Exception) {
            -1
        }
    }

    override fun isInterrupted(): Boolean {
        if (focusLost) return true
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return false
        val r = record ?: return false
        return try {
            r.activeRecordingConfiguration?.isClientSilenced == true
        } catch (_: Exception) {
            false
        }
    }

    override fun stop() {
        synchronized(lock) {
            stopped = true
            try {
                record?.stop()
            } catch (_: IllegalStateException) {
            }
        }
    }

    override fun close() {
        synchronized(lock) {
            stopped = true
            record?.let {
                try {
                    it.stop()
                } catch (_: IllegalStateException) {
                }
                it.release()
            }
            record = null
            focusRequest?.let { audioManager?.abandonAudioFocusRequest(it) }
            focusRequest = null
        }
    }

    private fun requestFocus() {
        val manager = audioManager ?: return
        val request = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setOnAudioFocusChangeListener { change ->
                if (change == AudioManager.AUDIOFOCUS_LOSS || change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT) {
                    focusLost = true
                }
            }
            .build()
        // Listening works without focus too; it only means other audio keeps playing
        manager.requestAudioFocus(request)
        focusRequest = request
    }

    companion object {
        private const val SAMPLE_RATE = WhisperTuning.SAMPLE_RATE
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        /** 200 ms of audio. */
        private const val BUFFER_BYTES = SAMPLE_RATE * 2 / 5
    }
}

// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import java.io.Closeable

/** 16 kHz mono 16-bit PCM input. */
interface AudioSource : Closeable {
    /** Starts capturing; false when the microphone cannot be opened (busy, no permission). */
    fun start(): Boolean

    /**
     * Blocking read of up to [length] samples into [buffer] at [offset].
     * Returns the number of samples read, 0 or less once stopped or on error.
     */
    fun read(buffer: ShortArray, offset: Int, length: Int): Int

    /**
     * True when the system feeds this client silence or took the input away (another app
     * recording, a phone call, audio focus lost).
     */
    fun isInterrupted(): Boolean

    /** Stops capturing and unblocks a pending [read]. Any thread, any number of times. */
    fun stop()

    /** Stops and releases the microphone. Call from the reading thread, after the last [read]. */
    override fun close()
}

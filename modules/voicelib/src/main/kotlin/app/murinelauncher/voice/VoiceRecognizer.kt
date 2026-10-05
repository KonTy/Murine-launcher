// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Thrown by a [SpeechEngineFactory] when loading the model would exhaust the device's memory. */
class InsufficientMemoryException(message: String) : IOException(message)

/** Thrown by a [SpeechEngineFactory] when no model is installed. */
class ModelMissingException : IOException("No speech model installed")

/**
 * One voice query: listen until the speaker stops, transcribe on-device, report the text.
 *
 * The model loads while the user speaks and is released as soon as the transcription is done, so
 * nothing stays in memory between queries. Two short-lived threads exist only for the duration of a
 * query (one reads the microphone, one loads and runs the model). Each query ends with exactly one
 * [Listener.onResult] or [Listener.onFailure], on [mainDispatcher], unless it was [cancel]led, after
 * which the listener hears nothing more. An instance is single use.
 */
class VoiceRecognizer(
    private val engineFactory: SpeechEngineFactory?,
    private val audioSourceFactory: () -> AudioSource,
    private val hasPermission: () -> Boolean,
    private val listener: Listener,
    mainDispatcher: CoroutineDispatcher,
    private val detectorConfig: EndpointDetector.Config = EndpointDetector.Config(),
    private val transcriptionTimeoutMs: Long = 30_000,
    private val executorFactory: () -> ExecutorService = {
        Executors.newFixedThreadPool(2) { r -> Thread(r, "MurineVoice") }
    },
) {
    enum class State { LISTENING, PROCESSING }

    enum class Failure {
        PERMISSION_DENIED, MODEL_MISSING, MODEL_FAILED, LOW_MEMORY,
        MIC_UNAVAILABLE, NO_SPEECH, TRANSCRIPTION_FAILED, TIMEOUT,
    }

    interface Listener {
        fun onStateChanged(state: State)
        fun onLevel(level: Float) {}
        fun onResult(text: String)
        fun onFailure(failure: Failure)
    }

    private enum class Stop { NONE, USER, CANCEL, ENGINE_FAILED }

    private enum class CaptureOutcome { SPEECH, NO_SPEECH, MIC_UNAVAILABLE, STOPPED }

    private class Captured(val samples: ShortArray, val start: Int, val end: Int, val outcome: CaptureOutcome)

    private sealed interface Loaded {
        class Ok(val engine: SpeechEngine) : Loaded
        class Failed(val failure: Failure) : Loaded
    }

    private val scope = CoroutineScope(SupervisorJob() + mainDispatcher)
    private val started = AtomicBoolean(false)
    private val finished = AtomicBoolean(false)
    private val engineRef = AtomicReference<SpeechEngine?>()
    private val sourceRef = AtomicReference<AudioSource?>()
    @Volatile private var stop = Stop.NONE
    @Volatile private var timedOut = false
    private var job: Job? = null

    /** True from [start] until the result, a failure or [cancel]. */
    val isActive: Boolean get() = started.get() && !finished.get()

    fun start() {
        if (!started.compareAndSet(false, true)) return
        if (!hasPermission()) return finishEarly(Failure.PERMISSION_DENIED)
        val factory = engineFactory ?: return finishEarly(Failure.MODEL_MISSING)

        listener.onStateChanged(State.LISTENING)
        job = scope.launch {
            val worker = executorFactory().asCoroutineDispatcher()
            try {
                coroutineScope {
                    val loaded = async(worker) { load(factory) }
                    val outcome = try {
                        pipeline(loaded, worker)
                    } catch (e: CancellationException) {
                        throw e
                    } catch (e: Exception) {
                        Failure.TRANSCRIPTION_FAILED
                    } catch (e: LinkageError) {
                        Failure.TRANSCRIPTION_FAILED
                    }
                    when (outcome) {
                        is String -> finish(text = outcome)
                        is Failure -> finish(outcome)
                    }
                    // Returns once the model load finished too, so cleanup below sees the engine
                }
            } finally {
                release(worker)
            }
        }
    }

    private fun finishEarly(failure: Failure) {
        finish(failure)
        scope.cancel()
    }

    /** Stops listening and transcribes what was said so far (the user tapped the microphone). */
    fun stopListening() {
        if (!isActive || stop != Stop.NONE) return
        stop = Stop.USER
        sourceRef.get()?.stop()
    }

    /** Abandons the query: stops the microphone, aborts the model, no further callbacks. */
    fun cancel() {
        if (!finished.compareAndSet(false, true)) return
        stop = Stop.CANCEL
        sourceRef.get()?.stop()
        engineRef.get()?.abort()
        job?.cancel() ?: scope.cancel()
    }

    private fun load(factory: SpeechEngineFactory): Loaded = try {
        val engine = factory.open()
        engineRef.set(engine)
        if (stop == Stop.CANCEL) engine.abort()
        Loaded.Ok(engine)
    } catch (e: InsufficientMemoryException) {
        failEarly()
        Loaded.Failed(Failure.LOW_MEMORY)
    } catch (e: ModelMissingException) {
        failEarly()
        Loaded.Failed(Failure.MODEL_MISSING)
    } catch (e: Exception) {
        failEarly()
        Loaded.Failed(Failure.MODEL_FAILED)
    } catch (e: LinkageError) {
        // A missing or broken native library must not take the launcher down
        failEarly()
        Loaded.Failed(Failure.MODEL_FAILED)
    }

    /** No point listening once the model is known not to load. */
    private fun failEarly() {
        if (stop == Stop.NONE) stop = Stop.ENGINE_FAILED
        sourceRef.get()?.stop()
    }

    /** Returns the transcript or a [Failure]. */
    private suspend fun CoroutineScope.pipeline(loaded: Deferred<Loaded>, worker: CoroutineDispatcher): Any {
        val captured = withContext(worker) { capture() }
        when (captured.outcome) {
            CaptureOutcome.MIC_UNAVAILABLE -> return Failure.MIC_UNAVAILABLE
            CaptureOutcome.NO_SPEECH -> return Failure.NO_SPEECH
            CaptureOutcome.STOPPED -> return (loaded.await() as? Loaded.Failed)?.failure ?: Failure.MIC_UNAVAILABLE
            CaptureOutcome.SPEECH -> Unit
        }
        if (!finished.get()) listener.onStateChanged(State.PROCESSING)

        val engine = when (val l = loaded.await()) {
            is Loaded.Failed -> return l.failure
            is Loaded.Ok -> l.engine
        }
        val watchdog = launch {
            delay(transcriptionTimeoutMs)
            timedOut = true
            engine.abort()
        }
        val raw = withContext(worker) {
            try {
                val count = maxOf(captured.end - captured.start, MIN_TRANSCRIBE_SAMPLES)
                // Zero-padded to Whisper's minimum input length
                val pcm = FloatArray(count)
                for (i in captured.start until captured.end) pcm[i - captured.start] = captured.samples[i] / 32768f
                engine.transcribe(pcm, count)
            } finally {
                // Free the model right away rather than when the UI is done with the result
                engineRef.getAndSet(null)?.close()
            }
        }
        watchdog.cancel()
        if (raw == null) return if (timedOut) Failure.TIMEOUT else Failure.TRANSCRIPTION_FAILED
        val text = TranscriptCleaner.clean(raw)
        return text.ifEmpty { Failure.NO_SPEECH }
    }

    /** Blocking capture loop; runs on a worker thread. */
    private fun capture(): Captured {
        val empty = ShortArray(0)
        if (stop != Stop.NONE) return stoppedBeforeListening()
        val source = try {
            audioSourceFactory()
        } catch (_: Exception) {
            return Captured(empty, 0, 0, CaptureOutcome.MIC_UNAVAILABLE)
        }
        sourceRef.set(source)
        try {
            // A stop that raced with the source being published or started
            if (stop != Stop.NONE) return stoppedBeforeListening()
            if (!source.start()) {
                return if (stop != Stop.NONE) stoppedBeforeListening()
                else Captured(empty, 0, 0, CaptureOutcome.MIC_UNAVAILABLE)
            }

            val frame = EndpointDetector.FRAME_SAMPLES
            val maxSamples = detectorConfig.maxUtteranceMs / EndpointDetector.FRAME_MS * frame
            val buffer = ShortArray(maxSamples)
            val detector = EndpointDetector(detectorConfig)
            var count = 0
            var frames = 0
            while (count + frame <= maxSamples) {
                var filled = 0
                while (filled < frame) {
                    val n = source.read(buffer, count + filled, frame - filled)
                    if (n <= 0) break
                    filled += n
                }
                when (stop) {
                    Stop.USER -> return userStopped(buffer, count + filled, detector)
                    Stop.CANCEL, Stop.ENGINE_FAILED -> return Captured(empty, 0, 0, CaptureOutcome.STOPPED)
                    Stop.NONE -> Unit
                }
                if (filled < frame) return Captured(empty, 0, 0, CaptureOutcome.MIC_UNAVAILABLE)

                val result = detector.feed(buffer, count, frame)
                count += frame
                frames++
                if (frames % LEVEL_EVERY_FRAMES == 0) postLevel(detector.level)
                if (frames % INTERRUPT_CHECK_FRAMES == 0 && source.isInterrupted()) {
                    return Captured(empty, 0, 0, CaptureOutcome.MIC_UNAVAILABLE)
                }
                when (result) {
                    EndpointDetector.Result.END_OF_SPEECH, EndpointDetector.Result.MAX_LENGTH ->
                        return speech(buffer, count, detector)
                    EndpointDetector.Result.NO_SPEECH -> return Captured(empty, 0, 0, CaptureOutcome.NO_SPEECH)
                    EndpointDetector.Result.DEAD_INPUT -> return Captured(empty, 0, 0, CaptureOutcome.MIC_UNAVAILABLE)
                    EndpointDetector.Result.WAITING, EndpointDetector.Result.SPEECH -> Unit
                }
            }
            return speech(buffer, count, detector)
        } finally {
            sourceRef.set(null)
            source.close()
        }
    }

    /** A tap before the microphone even started means nothing was said. */
    private fun stoppedBeforeListening() = Captured(ShortArray(0), 0, 0,
        if (stop == Stop.USER) CaptureOutcome.NO_SPEECH else CaptureOutcome.STOPPED)

    private fun userStopped(buffer: ShortArray, count: Int, detector: EndpointDetector): Captured {
        // Without a detected onset (a quiet speaker), still try whatever was recorded
        if (detector.speechStarted) return speech(buffer, count, detector)
        return if (count >= MIN_USER_STOP_SAMPLES) Captured(buffer, 0, count, CaptureOutcome.SPEECH)
        else Captured(ShortArray(0), 0, 0, CaptureOutcome.NO_SPEECH)
    }

    /** Trims silence around the speech (a shorter input is a faster transcription). */
    private fun speech(buffer: ShortArray, count: Int, detector: EndpointDetector): Captured {
        if (!detector.speechStarted) return Captured(ShortArray(0), 0, 0, CaptureOutcome.NO_SPEECH)
        val frame = EndpointDetector.FRAME_SAMPLES
        val start = maxOf(0, (detector.onsetFrame - LEAD_IN_FRAMES) * frame)
        val end = minOf(count, (detector.lastSpeechFrame + 1 + TAIL_FRAMES) * frame)
        return Captured(buffer, start, maxOf(start, end), CaptureOutcome.SPEECH)
    }

    private fun postLevel(level: Float) {
        scope.launch { if (!finished.get()) listener.onLevel(level) }
    }

    private fun finish(failure: Failure? = null, text: String? = null) {
        if (!finished.compareAndSet(false, true)) return
        if (text != null) listener.onResult(text) else listener.onFailure(failure!!)
    }

    private suspend fun release(worker: ExecutorCoroutineDispatcher) {
        withContext(NonCancellable) {
            engineRef.getAndSet(null)?.let { engine ->
                withContext(worker) {
                    try {
                        engine.close()
                    } catch (_: Exception) {
                    } catch (_: LinkageError) {
                    }
                }
            }
        }
        worker.close()
        scope.cancel()
    }

    companion object {
        private const val LEAD_IN_FRAMES = 15      // 300 ms before the onset
        private const val TAIL_FRAMES = 15         // 300 ms after the last speech
        private const val LEVEL_EVERY_FRAMES = 4   // 80 ms
        private const val INTERRUPT_CHECK_FRAMES = 25
        private const val MIN_USER_STOP_SAMPLES = WhisperTuning.SAMPLE_RATE * 3 / 10
        /** whisper.cpp refuses inputs shorter than 1 s. */
        private const val MIN_TRANSCRIBE_SAMPLES = WhisperTuning.SAMPLE_RATE + WhisperTuning.SAMPLE_RATE / 10
    }
}

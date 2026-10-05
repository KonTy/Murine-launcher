// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

/**
 * Drives the search box's microphone: one tap starts listening, a tap while listening stops and
 * transcribes, a tap while transcribing cancels. The recognised text only fills the search field,
 * which then shows the matching apps; searching the web stays an explicit action of the user.
 *
 * Main thread only.
 */
class VoiceSearchPresenter(
    private val box: SearchBox,
    private val recognizerFactory: (VoiceRecognizer.Listener) -> VoiceRecognizer,
) : VoiceRecognizer.Listener {

    enum class MicState { IDLE, LISTENING, PROCESSING }

    /** The search UI, as seen from voice input. */
    interface SearchBox {
        fun setMicState(state: MicState)
        fun setLevel(level: Float)
        /** Puts [text] in the search field, as if typed. */
        fun setQuery(text: String)
        fun showFailure(failure: VoiceRecognizer.Failure)
        /** The user's explicit search (IME action / enter). Voice input never calls it. */
        fun submit()
    }

    private var recognizer: VoiceRecognizer? = null

    var state = MicState.IDLE
        private set

    fun start() {
        if (state != MicState.IDLE) return
        val r = recognizerFactory(this)
        recognizer = r
        r.start()
    }

    fun onMicTapped() {
        when (state) {
            MicState.IDLE -> start()
            MicState.LISTENING -> recognizer?.stopListening()
            MicState.PROCESSING -> cancel()
        }
    }

    /** Stops everything without a result: the box was dismissed, home pressed, the launcher paused. */
    fun cancel() {
        recognizer?.cancel()
        recognizer = null
        setState(MicState.IDLE)
    }

    override fun onStateChanged(state: VoiceRecognizer.State) = setState(
        when (state) {
            VoiceRecognizer.State.LISTENING -> MicState.LISTENING
            VoiceRecognizer.State.PROCESSING -> MicState.PROCESSING
        }
    )

    override fun onLevel(level: Float) {
        if (state == MicState.LISTENING) box.setLevel(level)
    }

    override fun onResult(text: String) {
        recognizer = null
        setState(MicState.IDLE)
        box.setQuery(text)
    }

    override fun onFailure(failure: VoiceRecognizer.Failure) {
        recognizer = null
        setState(MicState.IDLE)
        box.showFailure(failure)
    }

    private fun setState(newState: MicState) {
        if (state == newState) return
        state = newState
        box.setMicState(newState)
    }
}

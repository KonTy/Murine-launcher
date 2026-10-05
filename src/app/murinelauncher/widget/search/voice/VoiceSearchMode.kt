package app.murinelauncher.widget.search.voice

/** What the search bar's microphone does. */
enum class VoiceSearchMode {
    /** Transcribes on this device with a downloaded speech model. */
    OFFLINE,

    /** Hands over to the system speech recognizer, which may be an online service. */
    SYSTEM,
}

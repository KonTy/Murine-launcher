// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

/** Turns raw model output into a search query. */
object TranscriptCleaner {
    // Non-speech annotations models emit for noise or music: [BLANK_AUDIO], (music), *laughs*, ♪
    private val ANNOTATIONS = Regex("""\[[^\]]*]|\([^)]*\)|\*[^*]*\*|[♪♫]+""")
    private val WHITESPACE = Regex("""\s+""")
    private const val EDGE_PUNCTUATION = ".,!?;:…¡¿\"'«»“”‘’„‚-–—"

    fun clean(raw: String): String = ANNOTATIONS.replace(raw, " ")
        .replace(WHITESPACE, " ")
        .trim { it.isWhitespace() || it in EDGE_PUNCTUATION }
}

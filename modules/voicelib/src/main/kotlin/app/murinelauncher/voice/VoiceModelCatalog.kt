// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

/**
 * A speech model published by FUTO for its voice input: OpenAI Whisper fine-tuned with FUTO's ACFT
 * method (dynamic audio context), converted to whisper.cpp GGML with Q8_0 weights.
 *
 * Murine never ships or mirrors these files: the user downloads them from FUTO's own server and
 * imports them. The source checkpoints are published by FUTO under Apache-2.0
 * (huggingface.co/futo-org/acft-whisper-*), fine-tuned from OpenAI Whisper (MIT).
 */
data class CatalogModel(
    /** Stable id, also the installed file name (without extension). */
    val id: String,
    val size: ModelSize,
    val multilingual: Boolean,
    /** Official download link from FUTO's model page (keyboard.futo.tech/voice-input-models). */
    val url: String,
    val bytes: Long,
    val sha256: String,
    /**
     * Approximate extra memory while loaded and transcribing, from whisper.cpp measurements
     * (tiny ~70 MB, base ~120 MB, small ~340 MB peak), rounded up.
     */
    val approxRamMb: Int,
) {
    val fileName: String get() = "$id.bin"
}

object VoiceModelCatalog {
    /** Where the user can read about the models and their terms. */
    const val MODELS_PAGE = "https://keyboard.futo.tech/voice-input-models"
    const val LICENSE_PAGE = "https://huggingface.co/futo-org/acft-whisper-base"

    private const val BASE_URL = "https://keyboard.futo.org/voice-input-"

    val models: List<CatalogModel> = listOf(
        CatalogModel(
            "futo-english-39", ModelSize.TINY, false, "${BASE_URL}english-39.bin", 43_550_795L,
            "4b5480aa1b14a7efc5b578ef176510970a898049671c3cd237285b3e3f6bfbfc", 80,
        ),
        CatalogModel(
            "futo-english-74", ModelSize.BASE, false, "${BASE_URL}english-74.bin", 81_781_811L,
            "e9b4b7b81b8a28769e8aa9962aa39bb9f21b622cf6a63982e93f065ed5caf1c8", 130,
        ),
        CatalogModel(
            "futo-english-244", ModelSize.SMALL, false, "${BASE_URL}english-244.bin", 264_477_561L,
            "58fbe949992dafed917590d58bc12ca577b08b9957f0b3e0d7ee71b64bed3aa8", 350,
        ),
        CatalogModel(
            "futo-multilingual-39", ModelSize.TINY, true, "${BASE_URL}multilingual-39.bin", 43_537_450L,
            "07aa4d514144deacf5ffec5cacb36c93dee272fda9e64ac33a801f8cd5cbd953", 80,
        ),
        CatalogModel(
            "futo-multilingual-74", ModelSize.BASE, true, "${BASE_URL}multilingual-74.bin", 81_768_602L,
            "e44f352c9aa2c3609dece20c733c4ad4a75c28cd9ab07d005383df55fa96efc4", 130,
        ),
        CatalogModel(
            "futo-multilingual-244", ModelSize.SMALL, true, "${BASE_URL}multilingual-244.bin", 264_464_624L,
            "15ef255465a6dc582ecf1ec651a4618c7ee2c18c05570bbe46493d248d465ac4", 350,
        ),
    )

    /** The default suggestion: accurate enough for app names, fast on mid-range phones. */
    val recommended: CatalogModel get() = models[1]

    fun byId(id: String): CatalogModel? = models.firstOrNull { it.id == id }

    fun bySha256(sha256: String): CatalogModel? =
        models.firstOrNull { it.sha256.equals(sha256, ignoreCase = true) }
}

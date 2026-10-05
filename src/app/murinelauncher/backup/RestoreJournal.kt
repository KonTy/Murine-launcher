/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: Apache-2.0
 */
package app.murinelauncher.backup

import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * A restart-only file transaction. Call before opening any of the managed files.
 * Originals remain beside their targets (including SQLite journals and preference .bak files)
 * until the restored workspace has loaded. Recovery never consumes its rollback source.
 */
internal class RestoreJournal(
    private val directory: File,
    private val targets: List<File>,
    private val syncDirectory: (File) -> Unit,
    private val move: (File, File) -> Unit,
) {
    private val state = File(directory, "state")
    private fun old(file: File) = File(file.path + ".restore-old")

    fun exists() = state.exists()
    fun phase() = state.readText()

    private fun transition(value: String) {
        write(File(directory, "state.tmp"), value.toByteArray())
        move(File(directory, "state.tmp"), state)
        syncDirectory(directory)
    }

    private fun write(file: File, bytes: ByteArray) {
        FileOutputStream(file).use { it.write(bytes); it.fd.sync() }
    }

    private fun ensureParent(file: File) {
        val parent = requireNotNull(file.parentFile)
        if (!parent.exists()) {
            if (!parent.mkdirs()) throw IOException("Cannot create restore directory")
            syncDirectory(requireNotNull(parent.parentFile))
        }
    }

    private fun copy(source: File, target: File) {
        ensureParent(target)
        val temp = File(target.path + ".restore-new")
        FileOutputStream(temp).use { output ->
            source.inputStream().use { it.copyTo(output) }
            output.fd.sync()
        }
        move(temp, target)
        syncDirectory(requireNotNull(target.parentFile))
    }

    private fun delete(file: File) {
        if (file.exists()) {
            if (!file.delete()) throw IOException("Cannot remove restore file")
            syncDirectory(requireNotNull(file.parentFile))
        }
    }

    /** The supplied map contains replacements; omitted targets are removed. */
    fun install(replacements: Map<File, File>) {
        check(!exists())
        check(targets.none { old(it).exists() })
        check(replacements.keys.all { it in targets })
        // Persist the presence map before the first rename, including originally absent files.
        write(File(directory, "present"), targets.joinToString("\n") {
            if (it.exists()) "1" else "0"
        }.toByteArray())
        syncDirectory(directory)
        transition("saving")
        targets.forEach { target ->
            if (target.exists()) {
                FileOutputStream(target, true).use { it.fd.sync() }
                move(target, old(target))
                syncDirectory(requireNotNull(target.parentFile))
            }
        }
        transition("installing")
        replacements.forEach { (target, source) -> copy(source, target) }
        transition("loading")
    }

    /** Idempotent even if recovery itself is interrupted. No SharedPreferences/SQLite handles. */
    fun rollback() {
        val phase = phase()
        check(phase in listOf("saving", "installing", "loading", "rolled-back"))
        if (phase == "rolled-back") return
        val present = File(directory, "present").readLines()
        check(present.size == targets.size && present.all { it == "0" || it == "1" })
        targets.forEachIndexed { index, target ->
            val saved = old(target)
            if (saved.exists()) {
                copy(saved, target)
            } else if (present[index] == "0") {
                delete(target)
            } else if (phase != "saving" || !target.exists()) {
                throw IOException("Restore rollback source is missing")
            }
        }
        transition("rolled-back")
    }

    fun commit() {
        check(phase() == "loading")
        transition("committed")
    }

    /** Only terminal transactions may discard their originals. Safe to retry after a restart. */
    fun cleanup() {
        check(phase() in listOf("committed", "rolled-back"))
        targets.forEach {
            delete(old(it))
            delete(File(it.path + ".restore-new"))
        }
    }
}

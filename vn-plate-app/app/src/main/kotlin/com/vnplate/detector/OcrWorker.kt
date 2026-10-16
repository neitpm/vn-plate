package com.vnplate.detector

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Request payload for asynchronous license plate recognition.
 */
data class OcrRequest(
    val frameId: Long,
    val bitmap: Bitmap,
    val box: RectF
)

/**
 * Result payload containing the recognized plate and associated metadata.
 */
data class OcrResult(
    val frameId: Long,
    val text: String,
    val latencyMs: Long
)

/**
 * Decoupled, non-blocking single-slot OCR worker.
 * Prevents camera preview / detector thread from stalling on ML Kit OCR execution.
 * Only the freshest frame request is processed; intermediate frames are dropped cleanly.
 */
class OcrWorker(private val plateOCR: PlateOCR = PlateOCR()) {

    companion object {
        private const val TAG = "OcrWorker"
    }

    private val executor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "OcrWorkerThread").apply { priority = Thread.NORM_PRIORITY - 1 }
    }

    private val isRunning = AtomicBoolean(false)
    private val pendingRequest = AtomicReference<OcrRequest?>(null)

    // Cached plate text and the frameId it was recognized from
    @Volatile
    var latestResult: OcrResult = OcrResult(frameId = -1L, text = "", latencyMs = 0L)
        private set

    /**
     * Submit an OCR task for a detected bounding box.
     * Uses single-slot replacement: if an OCR job is currently running,
     * the new request replaces any older pending request.
     */
    fun submit(frameId: Long, bitmap: Bitmap, box: RectF, onResult: ((OcrResult) -> Unit)? = null) {
        // Deep copy the crop if needed or copy the bitmap reference
        pendingRequest.set(OcrRequest(frameId, bitmap, RectF(box)))

        if (isRunning.compareAndSet(false, true)) {
            dispatchNext(onResult)
        }
    }

    private fun dispatchNext(onResult: ((OcrResult) -> Unit)?) {
        executor.execute {
            try {
                val req = pendingRequest.getAndSet(null)
                if (req != null) {
                    val t0 = System.currentTimeMillis()
                    val text = plateOCR.recognize(req.bitmap, req.box)
                    val latency = System.currentTimeMillis() - t0

                    if (text.isNotEmpty()) {
                        val result = OcrResult(req.frameId, text, latency)
                        latestResult = result
                        onResult?.invoke(result)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "OCR worker execution failed", e)
            } finally {
                isRunning.set(false)
                // Check if a newer request arrived while this one was executing
                if (pendingRequest.get() != null && isRunning.compareAndSet(false, true)) {
                    dispatchNext(onResult)
                }
            }
        }
    }

    fun close() {
        executor.shutdown()
        plateOCR.close()
    }
}

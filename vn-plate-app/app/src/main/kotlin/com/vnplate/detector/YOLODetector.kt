package com.vnplate.detector

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

data class Detection(
    val box      : RectF,
    val conf     : Float,
    val label    : String,
    val frameId  : Long = 0L,
    val plateText: String = ""   // ket qua OCR bien so
)

class YOLODetector(context: Context) {

    companion object {
        private const val MODEL_FILE  = "best.onnx"
        private const val INPUT_SIZE  = 224   // 224 nhanh ~2x so voi 320
        private const val CONF_THRESH = 0.50f
        private const val IOU_THRESH  = 0.50f
        private val LABELS = arrayOf("plate")
    }

    private val ortEnv  : OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session : OrtSession

    init {
        val modelFile = File(context.filesDir, MODEL_FILE)
        if (!modelFile.exists()) {
            context.assets.open(MODEL_FILE).use { input ->
                FileOutputStream(modelFile).use { output ->
                    val buf = ByteArray(8 * 1024 * 1024)
                    var len: Int
                    while (input.read(buf).also { len = it } != -1) {
                        output.write(buf, 0, len)
                    }
                }
            }
        }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = ortEnv.createSession(modelFile.absolutePath, opts)
    }

    /**
     * Detect bien so tu bitmap.
     * Chay trong background thread (blocking ONNX only).
     */
    fun detect(bitmap: Bitmap, frameId: Long = 0L): List<Detection> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        // ── 1. Letterbox transformation with exact aspect-ratio padding ──
        val lb = LetterboxUtil.letterbox(bitmap, INPUT_SIZE)
        val scaled = lb.bitmap

        // ── 2. Bitmap → CHW float buffer ──
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val buf  = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val rArr = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val gArr = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val bArr = FloatArray(INPUT_SIZE * INPUT_SIZE)
        for (i in pixels.indices) {
            val px = pixels[i]
            rArr[i] = ((px shr 16) and 0xFF) / 255f
            gArr[i] = ((px shr 8)  and 0xFF) / 255f
            bArr[i] = ( px         and 0xFF) / 255f
        }
        buf.put(rArr); buf.put(gArr); buf.put(bArr)
        buf.rewind()

        // ── 3. ONNX inference ──
        val shape       = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val inputName   = session.inputNames.iterator().next()
        val inputTensor = OnnxTensor.createTensor(ortEnv, buf, shape)
        val output      = session.run(mapOf(inputName to inputTensor))
        inputTensor.close()

        // ── 4. Parse YOLOv8 output [1, 5, N] ──
        val raw        = (output[0].value as Array<Array<FloatArray>>)[0]
        val numAnchors = raw[0].size
        val boxes      = mutableListOf<FloatArray>()

        for (i in 0 until numAnchors) {
            val conf = raw[4][i]
            if (conf < CONF_THRESH) continue
            val cx = raw[0][i]; val cy = raw[1][i]
            val w  = raw[2][i]; val h  = raw[3][i]
            boxes.add(floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf))
        }

        // ── 5. NMS ──
        val kept = nms(boxes)

        // ── 6. Map back via exact inverse letterbox transform ──
        return kept.map { b ->
            val box224 = RectF(b[0], b[1], b[2], b[3])
            val rect = LetterboxUtil.inverseTransform(
                box224 = box224,
                padLeft = lb.padLeft,
                padTop = lb.padTop,
                scaleRatio = lb.scaleRatio,
                origWidth = origW,
                origHeight = origH
            )
            Detection(
                box       = rect,
                conf      = b[4],
                label     = LABELS[0],
                frameId   = frameId,
                plateText = ""
            )
        }
    }

    private fun nms(boxes: List<FloatArray>): List<FloatArray> {
        val sorted = boxes.sortedByDescending { it[4] }.toMutableList()
        val result = mutableListOf<FloatArray>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { iou(best, it) > IOU_THRESH }
        }
        return result
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val ix1 = maxOf(a[0], b[0]); val iy1 = maxOf(a[1], b[1])
        val ix2 = minOf(a[2], b[2]); val iy2 = minOf(a[3], b[3])
        val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
        val aArea = (a[2] - a[0]) * (a[3] - a[1])
        val bArea = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aArea + bArea - inter + 1e-6f)
    }

    fun close() {
        session.close()
        ortEnv.close()
    }
}

package com.vnplate.detector

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class BoundingBoxView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var srcWidth   = 1f
    private var srcHeight  = 1f

    // ── Paints ───────────────────────────────────────────────
    private val boxPaint = Paint().apply {
        color       = Color.argb(220, 0, 230, 64)
        style       = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }
    private val fillPaint = Paint().apply {
        color = Color.argb(40, 0, 230, 64)
        style = Paint.Style.FILL
    }
    private val textBgPaint = Paint().apply {
        color = Color.argb(200, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val textPaint = Paint().apply {
        color       = Color.WHITE
        textSize    = 42f
        typeface    = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }
    private val platePaint = Paint().apply {
        color         = Color.WHITE
        textSize      = 52f
        typeface      = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        isAntiAlias   = true
        letterSpacing = 0.1f
    }
    private val plateBgPaint = Paint().apply {
        color = Color.argb(230, 0, 100, 200)
        style = Paint.Style.FILL
    }

    fun setDetections(list: List<Detection>, imageWidth: Int, imageHeight: Int) {
        detections = list
        srcWidth   = imageWidth.toFloat()
        srcHeight  = imageHeight.toFloat()
        invalidate()
    }

    fun clear() {
        detections = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        // ── Tinh toan offset/scale theo kieu fillCenter (giong PreviewView) ──
        // fillCenter: giu ty le anh, scale de day man hinh, crop phan du
        val viewW = width.toFloat()
        val viewH = height.toFloat()

        val scaleX = viewW / srcWidth
        val scaleY = viewH / srcHeight
        val scale  = maxOf(scaleX, scaleY)   // fillCenter dung scale lon hon

        // Offset de can giua (phan bi crop)
        val offsetX = (viewW - srcWidth  * scale) / 2f
        val offsetY = (viewH - srcHeight * scale) / 2f

        for (det in detections) {
            val left   = det.box.left   * scale + offsetX
            val top    = det.box.top    * scale + offsetY
            val right  = det.box.right  * scale + offsetX
            val bottom = det.box.bottom * scale + offsetY

            // Nen + khung
            canvas.drawRect(left, top, right, bottom, fillPaint)
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Label tren: "plate 86%"
            val confLabel = "${det.label} ${"%.0f".format(det.conf * 100)}%"
            val confW = textPaint.measureText(confLabel) + 16f
            val confH = textPaint.textSize + 8f
            canvas.drawRect(left, top - confH, left + confW, top, textBgPaint)
            canvas.drawText(confLabel, left + 8f, top - 6f, textPaint)

            // Bien so OCR phia duoi
            if (det.plateText.isNotEmpty()) {
                val plateW = platePaint.measureText(det.plateText) + 20f
                val plateH = platePaint.textSize + 10f
                val bgRect = RectF(left, bottom, left + plateW, bottom + plateH)
                canvas.drawRoundRect(bgRect, 6f, 6f, plateBgPaint)
                canvas.drawText(det.plateText, left + 10f, bottom + plateH - 6f, platePaint)
            }
        }
    }
}

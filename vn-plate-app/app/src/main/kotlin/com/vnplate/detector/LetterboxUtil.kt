package com.vnplate.detector

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF

/**
 * Result of letterbox transformation preserving exact Ultralytics YOLO aspect-ratio padding metadata.
 */
data class LetterboxResult(
    val bitmap: Bitmap,
    val scaleRatio: Float,
    val padLeft: Float,
    val padTop: Float,
    val padRight: Float,
    val padBottom: Float,
    val unpaddedWidth: Int,
    val unpaddedHeight: Int
)

/**
 * Ultralytics-compliant Letterbox Utility for YOLO models.
 * Replaces non-aspect-ratio preserving scaling with exact padding & symmetric rounding rules.
 */
object LetterboxUtil {

    private const val DEFAULT_BORDER_COLOR = 0xFF727272.toInt() // RGB(114, 114, 114)

    /**
     * Letterbox transform a Bitmap to targetSize x targetSize.
     * Matches Ultralytics letterbox implementation:
     * - scale ratio r = min(targetSize / height, targetSize / width)
     * - unpadded dimensions: round(width * r), round(height * r)
     * - padding split symmetrically with rounding (dh - 0.1 / dh + 0.1)
     * - pad area filled with RGB(114, 114, 114)
     */
    fun letterbox(
        bitmap: Bitmap,
        targetSize: Int = 224,
        borderColor: Int = DEFAULT_BORDER_COLOR
    ): LetterboxResult {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        // 1. Scale ratio
        val r = minOf(targetSize / origH, targetSize / origW)

        // 2. Unpadded dimensions
        val unpadW = Math.round(origW * r)
        val unpadH = Math.round(origH * r)

        // 3. Padding amounts
        val dw = (targetSize - unpadW) / 2f
        val dh = (targetSize - unpadH) / 2f

        // Ultralytics exact rounding
        val top = Math.round(dh - 0.1f)
        val bottom = Math.round(dh + 0.1f)
        val left = Math.round(dw - 0.1f)
        val right = Math.round(dw + 0.1f)

        // 4. Render into target bitmap
        val targetBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(targetBitmap)
        canvas.drawColor(borderColor)

        val srcRect = Rect(0, 0, bitmap.width, bitmap.height)
        val dstRect = Rect(left, top, left + unpadW, top + unpadH)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

        canvas.drawBitmap(bitmap, srcRect, dstRect, paint)

        return LetterboxResult(
            bitmap = targetBitmap,
            scaleRatio = r,
            padLeft = left.toFloat(),
            padTop = top.toFloat(),
            padRight = right.toFloat(),
            padBottom = bottom.toFloat(),
            unpaddedWidth = unpadW,
            unpaddedHeight = unpadH
        )
    }

    /**
     * Map bounding box coordinates from 224x224 input space back to original image space.
     * x_orig = (x_224 - padLeft) / scaleRatio
     * y_orig = (y_224 - padTop) / scaleRatio
     */
    fun inverseTransform(
        box224: RectF,
        padLeft: Float,
        padTop: Float,
        scaleRatio: Float,
        origWidth: Float? = null,
        origHeight: Float? = null
    ): RectF {
        var x1 = (box224.left - padLeft) / scaleRatio
        var y1 = (box224.top - padTop) / scaleRatio
        var x2 = (box224.right - padLeft) / scaleRatio
        var y2 = (box224.bottom - padTop) / scaleRatio

        if (origWidth != null && origHeight != null) {
            x1 = x1.coerceIn(0f, origWidth)
            y1 = y1.coerceIn(0f, origHeight)
            x2 = x2.coerceIn(0f, origWidth)
            y2 = y2.coerceIn(0f, origHeight)
        }

        return RectF(x1, y1, x2, y2)
    }

    /**
     * Forward map bounding box coordinates from original space to 224x224 space.
     */
    fun forwardTransform(
        boxOrig: RectF,
        padLeft: Float,
        padTop: Float,
        scaleRatio: Float
    ): RectF {
        val x1 = boxOrig.left * scaleRatio + padLeft
        val y1 = boxOrig.top * scaleRatio + padTop
        val x2 = boxOrig.right * scaleRatio + padLeft
        val y2 = boxOrig.bottom * scaleRatio + padTop
        return RectF(x1, y1, x2, y2)
    }
}

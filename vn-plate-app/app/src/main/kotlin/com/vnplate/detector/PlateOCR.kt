package com.vnplate.detector

import android.graphics.Bitmap
import android.graphics.RectF
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class PlateOCR {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    /**
     * Crop vung bien so tu bitmap goc, chay OCR, tra ve chuoi bien so da loc.
     * Ham nay chay dong bo (blocking) – goi trong background thread.
     */
    fun recognize(bitmap: Bitmap, box: RectF): String {
        // 1. Crop + padding them 10% de tranh cat mat chu
        val padX = box.width()  * 0.08f
        val padY = box.height() * 0.12f
        val left   = (box.left   - padX).coerceAtLeast(0f).toInt()
        val top    = (box.top    - padY).coerceAtLeast(0f).toInt()
        val right  = (box.right  + padX).coerceAtMost(bitmap.width.toFloat()).toInt()
        val bottom = (box.bottom + padY).coerceAtMost(bitmap.height.toFloat()).toInt()

        if (right <= left || bottom <= top) return ""

        val cropped = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)

        // 2. Scale len lon hon de ML Kit doc chinh xac hon
        val scaled = if (cropped.width < 300) {
            val scale = 300f / cropped.width
            Bitmap.createScaledBitmap(
                cropped,
                (cropped.width  * scale).toInt(),
                (cropped.height * scale).toInt(),
                true
            )
        } else cropped

        // 3. Chay ML Kit (async → convert sang blocking voi CountDownLatch)
        val latch  = CountDownLatch(1)
        var result = ""

        val image = InputImage.fromBitmap(scaled, 0)
        recognizer.process(image)
            .addOnSuccessListener { visionText ->
                result = cleanPlateText(visionText.text)
                latch.countDown()
            }
            .addOnFailureListener {
                latch.countDown()
            }

        // Timeout 1 giay, neu qua thi tra chuoi rong
        latch.await(1000, TimeUnit.MILLISECONDS)
        return result
    }

    /**
     * Loc va chuan hoa text bien so xe Viet Nam 2 hang:
     *   Hang tren: tinh/thanh pho + chu cai  (vd: 30A, 51F)
     *   Hang duoi: so seri                   (vd: 12345, 123.45)
     * Ket qua: "30A-12345" hoac "51F-123.45"
     */
    private fun cleanPlateText(raw: String): String {
        if (raw.isBlank()) return ""

        // Tach tung dong, loc ky tu hop le
        val lines = raw.lines()
            .map { line ->
                line.uppercase()
                    .replace(Regex("[^A-Z0-9\\-.]"), "")
                    .trim()
            }
            .filter { it.length >= 2 }

        if (lines.isEmpty()) return ""

        if (lines.size == 1) {
            // Chi 1 dong – tra ve neu du dai
            return if (lines[0].length >= 4) lines[0] else ""
        }

        // Nhieu dong: phan loai hang tren / hang duoi
        // Hang tren: co chu cai va it so (vd: "30A", "51F", "HN")
        // Hang duoi: chu yeu la so (vd: "12345", "123.45")
        val sortedByAlpha = lines.sortedByDescending { line ->
            line.count { it.isLetter() }
        }

        val top    = sortedByAlpha.first()                        // nhieu chu cai nhat → hang tren
        val bottom = lines.filter { it != top }
                          .maxByOrNull { it.length } ?: ""        // dai nhat → hang duoi

        return when {
            top.isNotEmpty() && bottom.isNotEmpty() -> "$top-$bottom"
            top.isNotEmpty()    -> top
            bottom.isNotEmpty() -> bottom
            else                -> ""
        }.also { result ->
            // Ket qua qua ngan → khong phai bien so
            if (result.replace("-", "").length < 4) return ""
        }
    }

    fun close() {
        recognizer.close()
    }
}

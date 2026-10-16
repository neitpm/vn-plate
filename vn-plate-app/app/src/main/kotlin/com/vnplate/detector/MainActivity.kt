package com.vnplate.detector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView : PreviewView
    private lateinit var boxView     : BoundingBoxView
    private lateinit var tvFps       : TextView
    private lateinit var tvCount     : TextView

    private lateinit var detector    : YOLODetector
    private val ocrWorker = OcrWorker()
    private val cameraExec: ExecutorService = Executors.newSingleThreadExecutor()

    private var frameCount  = 0L
    private var currentFrameId = 0L
    private var lastFpsTime = System.currentTimeMillis()
    private var currentFps  = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        boxView     = findViewById(R.id.boxView)
        tvFps       = findViewById(R.id.tvFps)
        tvCount     = findViewById(R.id.tvCount)

        tvCount.text = "Dang tai model..."
        cameraExec.execute {
            detector = YOLODetector(this)
            runOnUiThread {
                tvCount.text = "San sang!"
                if (hasCameraPermission()) startCamera()
                else requestCameraPermission()
            }
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // Lay rotation hien tai cua man hinh
            val rotation = windowManager.defaultDisplay.rotation

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetRotation(rotation)   // <-- truyen rotation vao day
                .build()

            analysis.setAnalyzer(cameraExec) { imageProxy ->
                processFrame(imageProxy)
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
            } catch (e: Exception) {
                Log.e("VnPlate", "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val frameId = ++currentFrameId
        // Lay goc xoay cua frame
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        var bitmap = imageProxy.toBitmap()
        imageProxy.close()

        // Xoay bitmap neu can (portrait = 90 do, landscape = 0)
        if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }

        val t0 = System.currentTimeMillis()
        val rawDetections = try {
            detector.detect(bitmap, frameId)
        } catch (e: Exception) {
            Log.e("VnPlate", "Detect error", e)
            emptyList()
        }
        val inferMs = System.currentTimeMillis() - t0

        // Asynchronously dispatch highest-confidence detection to OCR worker
        if (rawDetections.isNotEmpty()) {
            val bestDet = rawDetections.maxByOrNull { it.conf }!!
            ocrWorker.submit(frameId, bitmap, bestDet.box)
        }

        // Attach latest cached OCR result if valid
        val latestOcr = ocrWorker.latestResult
        val detections = rawDetections.map { det ->
            if (latestOcr.text.isNotEmpty()) {
                det.copy(plateText = latestOcr.text)
            } else {
                det
            }
        }

        // Cap nhat FPS
        frameCount++
        val now = System.currentTimeMillis()
        if (now - lastFpsTime >= 1000) {
            currentFps  = frameCount * 1000f / (now - lastFpsTime)
            frameCount  = 0
            lastFpsTime = now
        }

        val fps   = currentFps
        val count = detections.size
        val imgW  = bitmap.width
        val imgH  = bitmap.height

        runOnUiThread {
            boxView.setDetections(detections, imgW, imgH)
            val ocrStatus = if (latestOcr.latencyMs > 0) " | OCR: ${latestOcr.latencyMs}ms" else ""
            tvFps.text = "FPS: ${"%.1f".format(fps)}  |  Infer: ${inferMs}ms$ocrStatus"
            if (count > 0) {
                val plates = detections
                    .filter { it.plateText.isNotEmpty() }
                    .joinToString("  |  ") { it.plateText }
                tvCount.text = if (plates.isNotEmpty())
                    "Bien so: $plates"
                else
                    "Phat hien: $count bien so (dang doc...)"
            } else {
                tvCount.text = "Dang tim bien so..."
            }
        }
    }

    // ── Permissions ──────────────────────────────────────────
    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() =
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED)
            startCamera()
        else
            Toast.makeText(this, "Can quyen camera de su dung app!", Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExec.shutdown()
        ocrWorker.close()
        detector.close()
    }
}

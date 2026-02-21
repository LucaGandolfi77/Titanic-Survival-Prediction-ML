package com.example.mot

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * MainActivity – hosts the camera preview and tracking overlay.
 *
 * Flow:
 *   1. Request camera permission.
 *   2. Start CameraX preview.
 *   3. User taps & drags to select initial bbox → passed to TrackerEngine.
 *   4. Each frame analysed via CameraFragment's ImageAnalysis use-case.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var trackerEngine: TrackerEngine

    companion object {
        private const val REQUEST_CAMERA = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Simple layout: PreviewView fills the screen
        val previewView = PreviewView(this)
        setContentView(previewView)

        trackerEngine = TrackerEngine(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera(previewView)
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
        }
    }

    private fun startCamera(previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // Bind CameraX use-cases
            provider.unbindAll()
            provider.bindToLifecycle(this, cameraSelector, preview)

            // TODO: Add ImageAnalysis use-case for per-frame tracking via trackerEngine
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            val previewView = PreviewView(this)
            setContentView(previewView)
            startCamera(previewView)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        trackerEngine.close()
    }
}

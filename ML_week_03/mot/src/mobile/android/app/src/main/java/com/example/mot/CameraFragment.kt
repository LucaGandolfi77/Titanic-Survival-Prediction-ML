package com.example.mot

import android.graphics.RectF
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.atomic.AtomicBoolean

/**
 * CameraFragment – ImageAnalysis.Analyzer that feeds each frame
 * to TrackerEngine and publishes the resulting bbox.
 */
class CameraFragment(
    private val trackerEngine: TrackerEngine,
    private val onResult: (bbox: RectF, fps: Float) -> Unit,
) : ImageAnalysis.Analyzer {

    private val isTracking = AtomicBoolean(false)
    private var lastTs = System.nanoTime()

    /** Call once after user selects initial bbox. */
    fun initTracking(firstFrame: ImageProxy, bbox: RectF) {
        // Convert ImageProxy → Bitmap / ByteBuffer, pass to engine
        trackerEngine.initializeTracking(firstFrame, bbox)
        isTracking.set(true)
        firstFrame.close()
    }

    override fun analyze(image: ImageProxy) {
        if (!isTracking.get()) {
            image.close()
            return
        }

        val newBBox = trackerEngine.track(image)

        val now = System.nanoTime()
        val fps = 1e9f / (now - lastTs).coerceAtLeast(1).toFloat()
        lastTs = now

        onResult(newBBox, fps)
        image.close()
    }
}

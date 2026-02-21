package com.example.mot

import android.content.Context
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * TrackerEngine – TFLite inference wrapper for Android.
 *
 * Loads `siamese_tracker.tflite` from the assets folder and exposes
 * [initializeTracking] + [track] for per-frame Siamese tracking.
 */
class TrackerEngine(context: Context) {

    companion object {
        private const val MODEL_FILE = "siamese_tracker.tflite"
        private const val TEMPLATE_SIZE = 127
        private const val SEARCH_SIZE = 255
    }

    private var interpreter: Interpreter? = null
    private var currentBBox = RectF()

    // Pre-allocated input buffers
    private val templateBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * TEMPLATE_SIZE * TEMPLATE_SIZE * 3).apply {
            order(ByteOrder.nativeOrder())
        }
    private val searchBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * SEARCH_SIZE * SEARCH_SIZE * 3).apply {
            order(ByteOrder.nativeOrder())
        }

    init {
        val model = loadModelFile(context)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            // Uncomment for GPU: addDelegate(GpuDelegate())
        }
        interpreter = Interpreter(model, options)
    }

    // ── Public API ───────────────────────────────────────────────

    /** Initialise tracking with first-frame bbox (image coordinates). */
    fun initializeTracking(frame: ImageProxy, bbox: RectF) {
        currentBBox = RectF(bbox)
        // 1. Crop template patch from frame centred on bbox
        // 2. Resize to TEMPLATE_SIZE × TEMPLATE_SIZE
        // 3. Write normalised RGB floats into templateBuffer
    }

    /** Track target in current frame. Returns updated bbox. */
    fun track(frame: ImageProxy): RectF {
        // 1. Crop search region from frame around currentBBox
        // 2. Resize to SEARCH_SIZE × SEARCH_SIZE
        // 3. Write into searchBuffer
        // 4. Run interpreter
        // 5. Read response map, find peak, update currentBBox
        val inputs = arrayOf(templateBuffer, searchBuffer)
        // Output shape depends on model — adjust accordingly
        val responseMap = Array(1) { Array(17) { FloatArray(17) } }
        // interpreter?.runForMultipleInputsOutputs(inputs, mapOf(0 to responseMap))
        // ... find peak, convert to image coords, update currentBBox
        return currentBBox
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }

    // ── Helpers ──────────────────────────────────────────────────

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}

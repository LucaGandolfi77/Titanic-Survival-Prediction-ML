// TrackerEngine.swift – TFLite inference wrapper for iOS.
//
// Loads siamese_tracker.tflite from the app bundle and provides
// initializeTracking / track methods for per-frame object tracking.

import CoreGraphics
import CoreVideo
import Foundation

// Requires TensorFlowLiteSwift pod / SPM package
// import TensorFlowLite

final class TrackerEngine {

    // MARK: - Properties

    private let templateSize = 127
    private let searchSize = 255

    // TFLite interpreter (uncomment when TensorFlowLite is linked)
    // private var interpreter: Interpreter?

    /// Stored template embedding / patch (float buffer)
    private var templateBuffer: [Float] = []
    /// Last known bbox (x, y, w, h) in image coordinates
    private var currentBBox: CGRect = .zero

    // MARK: - Initialiser

    init() {
        guard let modelPath = Bundle.main.path(forResource: "siamese_tracker", ofType: "tflite") else {
            print("[TrackerEngine] ERROR: siamese_tracker.tflite not found in bundle.")
            return
        }
        print("[TrackerEngine] Model path: \(modelPath)")
        // interpreter = try? Interpreter(modelPath: modelPath)
        // try? interpreter?.allocateTensors()
    }

    // MARK: - Public API

    /// Initialise tracking with the first frame and user-selected bbox.
    func initializeTracking(frame: CVPixelBuffer, bbox: CGRect) {
        currentBBox = bbox
        // 1. Crop template patch from frame centred on bbox.
        // 2. Resize to (templateSize × templateSize).
        // 3. Normalise to [0,1] float32.
        // 4. Store in templateBuffer.
        print("[TrackerEngine] Initialised with bbox=\(bbox)")
    }

    /// Track the target in the given frame.
    /// - Returns: Updated bounding box in image coordinates.
    func track(frame: CVPixelBuffer) -> CGRect {
        // 1. Crop search patch centred on currentBBox (with context).
        // 2. Resize to (searchSize × searchSize).
        // 3. Normalise to [0,1] float32.
        // 4. Set template tensor + search tensor on interpreter.
        // 5. Invoke interpreter.
        // 6. Read response map output.
        // 7. Apply cosine window + find peak.
        // 8. Convert peak to new image-space bbox.
        // 9. Update currentBBox.
        return currentBBox
    }

    // MARK: - Helpers

    /// Convert CVPixelBuffer region to a normalised Float array.
    private func cropAndNormalise(
        _ buffer: CVPixelBuffer,
        roi: CGRect,
        targetSize: Int
    ) -> [Float] {
        // Lock base address, extract BGRA, convert to RGB float [0,1],
        // resize to targetSize × targetSize.
        // Return flat float array of length targetSize*targetSize*3.
        return Array(repeating: 0.0, count: targetSize * targetSize * 3)
    }
}

// CameraView.swift – AVFoundation camera preview wrapped for SwiftUI.
//
// Provides a live camera feed as a UIViewRepresentable that pushes
// CVPixelBuffers to a delegate for per-frame processing.

import AVFoundation
import SwiftUI
import UIKit

/// SwiftUI wrapper around AVCaptureVideoPreviewLayer.
struct CameraView: UIViewRepresentable {
    let session: AVCaptureSessionStub  // Replace with AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        // In production: add AVCaptureVideoPreviewLayer to view.layer
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

// MARK: - Full implementation guide (uncomment when integrating)
/*
 1. Create an AVCaptureSession.
 2. Add AVCaptureDeviceInput (back camera).
 3. Add AVCaptureVideoDataOutput, set sampleBufferDelegate.
 4. In delegate callback:
      - Convert CMSampleBuffer → CVPixelBuffer
      - Feed to TrackerEngine.track(frame:) on background queue
      - Publish result bbox to @Published property on main queue
 5. Display preview via AVCaptureVideoPreviewLayer.
*/

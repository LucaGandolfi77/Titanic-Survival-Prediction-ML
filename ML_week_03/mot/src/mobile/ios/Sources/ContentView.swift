// ContentView.swift – Main SwiftUI view for MotTrackerIOS
//
// Flow:
//   1. Camera preview fills the screen.
//   2. User taps + drags to draw a bounding box on the first freeze-frame.
//   3. Tracking starts; overlay shows bbox + FPS in real time.

import SwiftUI

struct ContentView: View {
    @StateObject private var cameraModel = CameraViewModel()
    @State private var isSelecting = false
    @State private var selectionRect: CGRect = .zero
    @State private var dragStart: CGPoint = .zero

    var body: some View {
        ZStack {
            // Live camera feed
            CameraView(session: cameraModel.session)
                .ignoresSafeArea()

            // Bounding-box overlay
            OverlayView(bbox: cameraModel.currentBBox, fps: cameraModel.fps)

            // Instruction text before tracking
            if !cameraModel.isTracking {
                VStack {
                    Spacer()
                    Text("Tap & drag to select target")
                        .font(.headline)
                        .padding()
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                    Spacer().frame(height: 80)
                }
            }

            // Selection rectangle drawn while dragging
            if isSelecting {
                Rectangle()
                    .stroke(Color.yellow, lineWidth: 2)
                    .frame(width: selectionRect.width, height: selectionRect.height)
                    .position(
                        x: selectionRect.midX,
                        y: selectionRect.midY
                    )
            }
        }
        .gesture(
            DragGesture(minimumDistance: 10)
                .onChanged { value in
                    if !cameraModel.isTracking {
                        isSelecting = true
                        dragStart = value.startLocation
                        let origin = CGPoint(
                            x: min(dragStart.x, value.location.x),
                            y: min(dragStart.y, value.location.y)
                        )
                        selectionRect = CGRect(
                            origin: origin,
                            size: CGSize(
                                width: abs(value.location.x - dragStart.x),
                                height: abs(value.location.y - dragStart.y)
                            )
                        )
                    }
                }
                .onEnded { _ in
                    if isSelecting {
                        isSelecting = false
                        cameraModel.startTracking(bbox: selectionRect)
                    }
                }
        )
        .onAppear {
            cameraModel.configure()
        }
    }
}

// MARK: - Camera ViewModel (placeholder — wire to TrackerEngine)

class CameraViewModel: ObservableObject {
    @Published var isTracking = false
    @Published var currentBBox: CGRect = .zero
    @Published var fps: Double = 0.0

    let session = AVCaptureSessionStub()

    func configure() {
        // TODO: Start AVCaptureSession, hook frame delegate
        print("[CameraViewModel] configure() called — start camera session here.")
    }

    func startTracking(bbox: CGRect) {
        isTracking = true
        currentBBox = bbox
        print("[CameraViewModel] startTracking bbox=\(bbox)")
        // TODO: Call TrackerEngine.initializeTracking(frame:bbox:)
    }
}

/// Stub — replace with real AVCaptureSession wrapper.
class AVCaptureSessionStub {}

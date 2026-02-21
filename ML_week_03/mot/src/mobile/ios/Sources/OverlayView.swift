// OverlayView.swift – Draw bounding box + FPS overlay on camera feed.

import SwiftUI

struct OverlayView: View {
    var bbox: CGRect
    var fps: Double

    var body: some View {
        ZStack {
            // Bounding box
            if bbox != .zero {
                Rectangle()
                    .stroke(Color.green, lineWidth: 3)
                    .frame(width: bbox.width, height: bbox.height)
                    .position(x: bbox.midX, y: bbox.midY)
            }

            // FPS label
            VStack {
                HStack {
                    Text(String(format: "FPS: %.1f", fps))
                        .font(.system(.caption, design: .monospaced))
                        .padding(6)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                    Spacer()
                }
                .padding(.horizontal)
                Spacer()
            }
            .padding(.top, 50)
        }
    }
}

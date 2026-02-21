# Lightweight Mobile Object Tracker (MOT)

A **Siamese-network single-object tracker** optimised for real-time inference on Apple Silicon M1, iOS and Android — built from scratch with **TensorFlow / Keras** and deployed via **TensorFlow Lite + OpenCV**.

---

## Architecture

```
┌──────────────┐        ┌──────────────┐
│  Template     │        │   Search      │
│  127×127×3    │        │   255×255×3   │
└──────┬───────┘        └──────┬───────┘
       │                       │
       ▼                       ▼
  ┌─────────────────────────────────┐
  │   Shared Backbone (4-block      │
  │   depthwise-separable CNN)      │
  │   MobileNet-style, ReLU6, BN   │
  └──────────┬──────────┬──────────┘
             │          │
        feat_t      feat_s
             │          │
             ▼          ▼
       ┌─────────────────────┐
       │  Cross-Correlation   │
       │  (depth-wise conv)   │
       └──────────┬──────────┘
                  │
           response map (H×W×1)
                  │
           ┌──────┴──────┐
           │ Find peak →  │
           │ update bbox  │
           └─────────────┘
```

**Key specs:**

| Property | Value |
|----------|-------|
| Backbone | 4-block depthwise-separable CNN |
| Template size | 127 × 127 px |
| Search size | 255 × 255 px |
| Activations | ReLU6 |
| Correlation | Depth-wise cross-correlation |
| Loss | Balanced logistic (BCE) |
| Target model size | < 5–10 MB (TFLite FP16) |
| Target latency (M1) | < 10 ms / frame |

---

## Project Structure

```
mot/
├── configs/                  # YAML config files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── tracker_config.yaml
├── data/
│   ├── raw/                  # OTB-like video sequences
│   ├── processed/            # template/search patch pairs
│   └── annotations/          # initial bbox JSON files
├── models/
│   ├── siamese_tracker_tf/   # Keras checkpoints
│   ├── siamese_tracker.tflite
│   └── label_map.json
├── src/
│   ├── training/             # model def, dataset, losses, train, export
│   ├── tracking/             # tracker core, TFLite wrapper, OpenCV I/O
│   ├── demo/                 # desktop demo + bbox selector
│   └── mobile/
│       ├── ios/              # Swift/SwiftUI demo app
│       └── android/          # Kotlin + CameraX demo app
├── notebooks/                # architecture viz, training logs, M1 benchmark
├── scripts/                  # data prep, benchmark, CoreML conversion
├── Makefile
├── requirements.txt
├── setup.py
└── README.md
```

---

## Setup (macOS M1)

```bash
cd ML_week_03/mot

# Create virtual environment & install deps
make setup

# Activate
source .venv/bin/activate
```

---

## Training

### 1. Prepare data

Place video sequences under `data/raw/<sequence_name>/`:
```
data/raw/Basketball/
  imgs/       # 001.jpg, 002.jpg, …
  groundtruth.txt   # x,y,w,h per frame
```

Then:
```bash
make data
```

> If no data is available, training falls back to **synthetic pairs** for smoke-testing.

### 2. Train

```bash
make train
```

Monitors with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

### 3. Export to TFLite

```bash
make export
```

This produces `models/siamese_tracker.tflite` (FP16 quantised).

---

## Desktop Demo

```bash
make demo
```

- Opens webcam (or pass `--source path/to/video.mp4`)
- Draw a bounding box on the first frame
- Real-time tracking with FPS overlay
- Press **ESC** to quit

---

## Benchmark

```bash
make benchmark
```

Runs 500 synthetic frames through TFLite and prints latency stats.

---

## Mobile Apps

### iOS (Swift / SwiftUI)

1. Copy `models/siamese_tracker.tflite` into `src/mobile/ios/Resources/`.
2. Open `MotTrackerIOS.xcodeproj` in Xcode.
3. Add `TensorFlowLiteSwift` via SPM.
4. Build & run on device or simulator.

### Android (Kotlin / CameraX)

1. Copy `models/siamese_tracker.tflite` into `src/mobile/android/app/src/main/assets/`.
2. Open `src/mobile/android/` in Android Studio.
3. Sync Gradle and run on device.

---

## Performance

| Platform | Accuracy (OTB-50) | FPS (M1) | FPS (iPhone 13) | Notes |
|----------|-------------------|----------|-----------------|-------|
| TFLite FP16 | TBD | TBD | TBD | XNNPACK delegate |
| TFLite INT8 | TBD | TBD | TBD | Dynamic range quant |
| Keras FP32 | TBD | TBD | — | Baseline reference |

> Fill in after training on a real dataset.

---

## Limitations & Possible Extensions

- **Single-object only** — extend to multi-object tracking (MOT) with a detector for re-initialisation.
- **No re-detection** — if the target is lost, tracking drifts. Add a re-detection module triggered by low confidence.
- **Fixed scale** — the current tracker doesn't handle large scale changes. Add multi-scale search or scale estimation head.
- **Backbone** — swap the lightweight CNN for a MobileNetV3 or EfficientNet-Lite backbone for better accuracy.
- **Re-identification** — combine with a ReID head for tracking through occlusions.
- **CoreML** — run `make coreml` for native iOS performance with the Metal GPU.
- **On-device training** — explore federated fine-tuning for domain adaptation.

---

## License

MIT

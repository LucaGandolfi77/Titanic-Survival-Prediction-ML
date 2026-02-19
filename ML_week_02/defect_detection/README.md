# Real-Time Defect Detection System

> **Production-ready PCB defect detection** with YOLOv8 fine-tuning, REST API, WebSocket streaming, and interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

End-to-end system for detecting manufacturing defects on PCB boards using state-of-the-art YOLOv8 object detection. The pipeline covers **5 stages**:

| Stage | Description |
|-------|-------------|
| **1. Dataset Preparation** | Synthetic data generation, format conversion (COCO/VOC → YOLO), stratified splitting, augmentation |
| **2. Model Training** | YOLOv8 fine-tuning (nano/small/medium) with configurable hyperparameters and Optuna hyperopt |
| **3. Evaluation** | mAP50/95, per-class AP, confusion matrices, confidence analysis |
| **4. Inference Pipeline** | Single-image, batch, video, and live webcam detection with FPS tracking |
| **5. Deployment** | FastAPI REST + WebSocket, Streamlit dashboard, Gradio demo, Docker containers |

### Defect Classes (5)

| ID | Name | Description |
|----|------|-------------|
| 0 | `scratch` | Linear surface scratches |
| 1 | `dent` | Circular depressions |
| 2 | `discoloration` | Colour-shifted oxidation patches |
| 3 | `crack` | Branching fracture lines |
| 4 | `missing_component` | Empty component pads |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset + split
bash scripts/download_dataset.sh

# 3. Train YOLOv8s for 50 epochs on CPU
bash scripts/train.sh s 50 16 cpu

# 4. Launch API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 5. Launch Streamlit dashboard
streamlit run src/ui/streamlit_app.py --server.port 8503
```

Or use **Make**:

```bash
make install      # install deps
make data         # generate & split dataset
make train        # train yolov8s
make api          # start FastAPI
make ui           # start Streamlit
make test         # run pytest
make all          # full pipeline
```

---

## Project Structure

```
defect_detection/
├── configs/
│   ├── training/         # base.yaml, augmentation.yaml, hyperopt.yaml
│   ├── model/            # yolov8n.yaml, yolov8s.yaml, yolov8m.yaml
│   └── deployment/       # api_config.yaml, inference_config.yaml
├── data/
│   ├── dataset.yaml      # YOLO dataset descriptor
│   ├── raw/              # original images
│   ├── processed/        # train/val/test splits
│   └── synthetic/        # generated PCB images
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.training
│   └── docker-compose.yml
├── models/
│   ├── trained/          # .pt weights
│   └── exported/         # ONNX / CoreML exports
├── notebooks/            # Jupyter analysis notebooks
├── scripts/
│   ├── download_dataset.sh
│   ├── train.sh
│   ├── export_model.sh
│   └── benchmark.py
├── src/
│   ├── api/              # FastAPI (main, routes, schemas, websocket)
│   ├── data/             # Data pipeline (synthetic, augment, split, convert)
│   ├── evaluation/       # Metrics, confusion matrix, visualisation
│   ├── inference/        # Predictor, batch, video, webcam, API helpers
│   ├── models/           # Trainer, exporter, ensemble
│   ├── ui/               # Streamlit + Gradio apps
│   └── utils/            # Config, logging, visualisation helpers
├── tests/                # pytest suite
├── requirements.txt
├── setup.py
├── pyproject.toml
└── Makefile
```

---

## Configuration

All settings live in YAML files under `configs/`:

- **`training/base.yaml`** — epochs, batch size, learning rate, augmentation params
- **`training/augmentation.yaml`** — Albumentations pipeline definition
- **`training/hyperopt.yaml`** — Optuna search space for hyperparameter optimisation
- **`model/yolov8{n,s,m}.yaml`** — model-size-specific configurations
- **`deployment/api_config.yaml`** — FastAPI host, port, CORS, upload limits
- **`deployment/inference_config.yaml`** — confidence/IoU thresholds, colour map

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Single image detection |
| `POST` | `/predict/batch` | Multi-image detection |
| `POST` | `/predict/video` | Frame-by-frame video analysis |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Cumulative prediction stats |

### WebSocket

```
ws://localhost:8000/ws/stream
```

Send binary JPEG frames → receive JSON detection results in real-time.

---

## Docker

```bash
# API + UI
cd docker && docker-compose up -d

# Training (one-shot)
docker-compose --profile training up trainer
```

---

## Testing

```bash
pytest tests/ -v --tb=short
```

Test suite covers:
- **`test_data_loader.py`** — Synthetic generation, dataset splitting, annotation conversion
- **`test_augmentation.py`** — Pipeline construction, image+bbox augmentation, offline augment I/O
- **`test_inference.py`** — Detection dataclasses, IoU computation, mAP evaluation, confusion matrix
- **`test_api.py`** — Pydantic schemas, FastAPI routes with mocked predictor

---

## Benchmarking

```bash
python scripts/benchmark.py --device cpu --models yolov8n.pt yolov8s.pt --iterations 100
```

Reports mean latency, FPS, P50/P95/P99 for each model.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Detection | Ultralytics YOLOv8 |
| Deep Learning | PyTorch 2.1+ |
| Export | ONNX Runtime, CoreML Tools |
| Augmentation | Albumentations |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit 1.30+ |
| Demo | Gradio 4.14+ |
| Visualisation | Plotly, OpenCV |
| Logging | Loguru |
| Schemas | Pydantic v2 |
| Containers | Docker + Compose |

---

## License

MIT

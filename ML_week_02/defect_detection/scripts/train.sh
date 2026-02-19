#!/usr/bin/env bash
# train.sh — Train YOLOv8 on the PCB defect dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL_SIZE="${1:-s}"       # n, s, m, l, x
EPOCHS="${2:-100}"
BATCH="${3:-16}"
DEVICE="${4:-cpu}"

echo "╔══════════════════════════════════════════╗"
echo "║   YOLOv8 Training — Defect Detection     ║"
echo "╚══════════════════════════════════════════╝"
echo "  Model  : yolov8${MODEL_SIZE}"
echo "  Epochs : ${EPOCHS}"
echo "  Batch  : ${BATCH}"
echo "  Device : ${DEVICE}"
echo ""

python -c "
from src.models.yolo_trainer import YOLOv8Trainer
from pathlib import Path

trainer = YOLOv8Trainer(model_size='${MODEL_SIZE}', pretrained=True, device='${DEVICE}')
results = trainer.train(
    data_yaml=Path('data/dataset.yaml'),
    config={'epochs': ${EPOCHS}, 'batch': ${BATCH}, 'imgsz': 640},
    name='defect_${MODEL_SIZE}_e${EPOCHS}',
)
print('Done!')
"

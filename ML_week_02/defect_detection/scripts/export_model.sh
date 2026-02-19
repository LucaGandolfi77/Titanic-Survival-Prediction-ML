#!/usr/bin/env bash
# export_model.sh — Export trained model to ONNX / CoreML
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

WEIGHTS="${1:-models/trained/best.pt}"
FORMAT="${2:-onnx}"

echo "╔══════════════════════════════════════════╗"
echo "║   Model Export                            ║"
echo "╚══════════════════════════════════════════╝"
echo "  Weights : ${WEIGHTS}"
echo "  Format  : ${FORMAT}"
echo ""

python -c "
from src.models.model_exporter import export_model
from pathlib import Path
export_model(Path('${WEIGHTS}'), fmt='${FORMAT}', imgsz=640, half=False)
"

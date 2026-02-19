#!/usr/bin/env bash
# download_dataset.sh — Download and prepare PCB defect datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

echo "╔══════════════════════════════════════════╗"
echo "║   PCB Defect Dataset Downloader          ║"
echo "╚══════════════════════════════════════════╝"

mkdir -p "$DATA_DIR/raw/images" "$DATA_DIR/raw/labels"

# Option 1: Generate synthetic dataset
echo ""
echo "Generating synthetic PCB defect dataset …"
cd "$PROJECT_DIR"
python -c "
from src.data.synthetic_generator import SyntheticDefectGenerator
from pathlib import Path
gen = SyntheticDefectGenerator(output_dir=Path('data/synthetic'), seed=42)
gen.generate_dataset(n_images=2000, defects_per_image=(1, 4))
"

echo ""
echo "✓ Synthetic dataset generated in data/synthetic/"
echo ""

# Split into train/val/test
echo "Splitting dataset …"
python -c "
from src.data.dataset_splitter import stratified_split
from pathlib import Path
stratified_split(
    images_dir=Path('data/synthetic/images'),
    labels_dir=Path('data/synthetic/labels'),
    output_dir=Path('data/processed'),
    ratios=(0.7, 0.2, 0.1),
    seed=42,
)
"
echo "✓ Dataset split complete!"
echo ""
echo "Dataset structure:"
find "$DATA_DIR/processed" -type f | head -20
echo "..."
echo ""
echo "Image counts:"
echo "  train: $(ls "$DATA_DIR/processed/train/images" 2>/dev/null | wc -l)"
echo "  val:   $(ls "$DATA_DIR/processed/val/images" 2>/dev/null | wc -l)"
echo "  test:  $(ls "$DATA_DIR/processed/test/images" 2>/dev/null | wc -l)"

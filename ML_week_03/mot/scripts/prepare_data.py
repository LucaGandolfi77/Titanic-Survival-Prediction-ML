"""
prepare_data.py – Slice video sequences into (template, search) patch pairs.

Reads from ``data/raw/<sequence>/`` and writes processed NumPy arrays
(or image files) to ``data/processed/``.

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --max-pairs 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]  # mot/

# Reuse the dataset helpers
import sys
sys.path.insert(0, str(ROOT))
from src.training.dataset import _crop_and_resize, _parse_groundtruth


def prepare(max_pairs: int = 50) -> None:
    cfg_path = ROOT / "configs" / "training_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    dcfg = cfg["data"]
    template_size: int = dcfg["template_size"]
    search_size: int = dcfg["search_size"]
    context = 0.5

    raw_dir = ROOT / "data" / "raw"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"[prepare_data] {raw_dir} does not exist — nothing to do.")
        print("  Place video sequences under data/raw/<sequence>/imgs/ + groundtruth.txt")
        return

    total = 0
    for seq_dir in sorted(raw_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        imgs_dir = seq_dir / "imgs"
        gt_path = seq_dir / "groundtruth.txt"
        if not imgs_dir.exists() or not gt_path.exists():
            continue

        frames = sorted(imgs_dir.glob("*.*"))
        bboxes = _parse_groundtruth(gt_path)
        n = min(len(frames), len(bboxes))
        if n < 2:
            continue

        seq_out = out_dir / seq_dir.name
        seq_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for idx in range(min(n, max_pairs)):
            search_idx = min(idx + 1, n - 1)
            img_t = cv2.imread(str(frames[idx]))
            img_s = cv2.imread(str(frames[search_idx]))
            if img_t is None or img_s is None:
                continue

            t_patch = _crop_and_resize(img_t, bboxes[idx], template_size, context)
            s_patch = _crop_and_resize(img_s, bboxes[search_idx], search_size, context)

            cv2.imwrite(str(seq_out / f"{count:05d}_template.jpg"), t_patch)
            cv2.imwrite(str(seq_out / f"{count:05d}_search.jpg"), s_patch)
            count += 1

        total += count
        print(f"  {seq_dir.name}: {count} pairs")

    print(f"\n[prepare_data] Total pairs written: {total}")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=50)
    args = parser.parse_args()
    prepare(args.max_pairs)

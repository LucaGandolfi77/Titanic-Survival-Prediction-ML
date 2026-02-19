"""
annotation_converter.py – Convert between annotation formats.

Supported conversions:
  COCO JSON  →  YOLO txt
  Pascal VOC XML  →  YOLO txt
  YOLO txt  →  COCO JSON  (for evaluation libraries)
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── COCO → YOLO ─────────────────────────────────────────────

def coco_to_yolo(
    coco_json: Path,
    output_dir: Path,
    class_mapping: Optional[Dict[str, int]] = None,
) -> int:
    """Convert a COCO-format JSON annotation file to per-image YOLO txt files.

    Returns the number of label files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json) as fh:
        coco = json.load(fh)

    # Build image id → filename map
    id_to_file = {img["id"]: img for img in coco["images"]}

    # Category mapping
    if class_mapping is None:
        class_mapping = {cat["name"]: cat["id"] for cat in coco["categories"]}
    cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

    # Group annotations by image
    img_anns: Dict[int, list] = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    count = 0
    for img_id, anns in img_anns.items():
        info = id_to_file[img_id]
        w, h = info["width"], info["height"]
        stem = Path(info["file_name"]).stem

        lines: List[str] = []
        for ann in anns:
            cls_idx = cat_id_to_idx.get(ann["category_id"])
            if cls_idx is None:
                continue
            bx, by, bw, bh = ann["bbox"]  # COCO: x, y, w, h (absolute)
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (output_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        count += 1

    logger.info(f"COCO → YOLO: wrote {count} label files to {output_dir}")
    return count


# ── Pascal VOC → YOLO ───────────────────────────────────────

def voc_to_yolo(
    voc_dir: Path,
    output_dir: Path,
    class_names: List[str],
) -> int:
    """Convert Pascal VOC XML annotations to YOLO txt.

    *class_names* defines the label-to-index mapping (order = index).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    count = 0
    for xml_path in sorted(Path(voc_dir).glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        lines: List[str] = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            idx = name_to_idx.get(name)
            if idx is None:
                continue
            bb = obj.find("bndnbox") or obj.find("bndbox")
            if bb is None:
                continue
            xmin = float(bb.find("xmin").text)
            ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text)
            ymax = float(bb.find("ymax").text)
            cx = (xmin + xmax) / 2 / w
            cy = (ymin + ymax) / 2 / h
            nw = (xmax - xmin) / w
            nh = (ymax - ymin) / h
            lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (output_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines) + "\n")
        count += 1

    logger.info(f"VOC → YOLO: wrote {count} label files to {output_dir}")
    return count


# ── YOLO → COCO ─────────────────────────────────────────────

def yolo_to_coco(
    images_dir: Path,
    labels_dir: Path,
    class_names: List[str],
    output_json: Path,
) -> Path:
    """Convert YOLO txt + images to a COCO-format JSON file."""
    import cv2

    images_list, annotations, categories = [], [], []
    for i, name in enumerate(class_names):
        categories.append({"id": i, "name": name})

    ann_id = 1
    for img_id, img_path in enumerate(sorted(Path(images_dir).glob("*.jpg")), start=1):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        images_list.append({
            "id": img_id, "file_name": img_path.name, "width": w, "height": h,
        })
        lbl_path = Path(labels_dir) / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            cls_idx = int(parts[0])
            cx, cy, nw, nh = [float(x) for x in parts[1:5]]
            bw, bh = nw * w, nh * h
            bx, by = cx * w - bw / 2, cy * h - bh / 2
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cls_idx,
                "bbox": [bx, by, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images_list, "annotations": annotations, "categories": categories}
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(coco, fh, indent=2)
    logger.info(f"YOLO → COCO: wrote {output_json}")
    return output_json

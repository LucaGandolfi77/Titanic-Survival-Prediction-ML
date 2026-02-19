"""
test_data_loader.py – Tests for dataset handling: synthetic generation, splitting, conversion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# ensure project root is on path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dataset(tmp_path: Path):
    """Create a tiny synthetic dataset in a temp dir."""
    from src.data.synthetic_generator import SyntheticDefectGenerator

    gen = SyntheticDefectGenerator(output_dir=tmp_path / "syn", image_size=(128, 128), seed=42)
    gen.generate_dataset(n_images=20, defects_per_image=(1, 3))
    return tmp_path / "syn"


@pytest.fixture()
def yolo_label_dir(tmp_path: Path):
    """Write a few minimal YOLO-format label files."""
    lbl_dir = tmp_path / "labels"
    lbl_dir.mkdir()
    for i in range(10):
        cls_id = i % 5
        (lbl_dir / f"img_{i:04d}.txt").write_text(
            f"{cls_id} 0.5 0.5 0.2 0.3\n"
        )
    return lbl_dir


@pytest.fixture()
def image_and_label_dirs(tmp_path: Path, yolo_label_dir: Path):
    """10 tiny images + matching labels."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:04d}.jpg"), img)
    return img_dir, yolo_label_dir


# ─────────────────────────────────────────────────────────────
# SyntheticDefectGenerator
# ─────────────────────────────────────────────────────────────

class TestSyntheticGenerator:
    def test_generates_correct_count(self, tmp_dataset: Path):
        images = list((tmp_dataset / "images").glob("*.jpg"))
        labels = list((tmp_dataset / "labels").glob("*.txt"))
        assert len(images) == 20
        assert len(labels) == 20

    def test_label_format(self, tmp_dataset: Path):
        label = next((tmp_dataset / "labels").glob("*.txt"))
        lines = label.read_text().strip().split("\n")
        for line in lines:
            parts = line.split()
            assert len(parts) == 5, f"Expected 5 fields, got {len(parts)}"
            cls_id = int(parts[0])
            assert 0 <= cls_id <= 4
            for v in parts[1:]:
                f = float(v)
                assert 0.0 <= f <= 1.0

    def test_image_size(self, tmp_dataset: Path):
        img = cv2.imread(str(next((tmp_dataset / "images").glob("*.jpg"))))
        assert img.shape[:2] == (128, 128)

    def test_class_names_attribute(self):
        from src.data.synthetic_generator import SyntheticDefectGenerator
        assert 0 in SyntheticDefectGenerator.CLASS_NAMES
        assert SyntheticDefectGenerator.CLASS_NAMES[4] == "missing_component"

    def test_idempotent_with_same_seed(self, tmp_path: Path):
        from src.data.synthetic_generator import SyntheticDefectGenerator

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        for d in (dir_a, dir_b):
            gen = SyntheticDefectGenerator(output_dir=d, image_size=(64, 64), seed=99)
            gen.generate_dataset(n_images=5, defects_per_image=(1, 2))
        labels_a = sorted((dir_a / "labels").glob("*.txt"))
        labels_b = sorted((dir_b / "labels").glob("*.txt"))
        for la, lb in zip(labels_a, labels_b):
            assert la.read_text() == lb.read_text()


# ─────────────────────────────────────────────────────────────
# Dataset splitter
# ─────────────────────────────────────────────────────────────

class TestDatasetSplitter:
    def test_split_counts(self, image_and_label_dirs, tmp_path: Path):
        from src.data.dataset_splitter import stratified_split

        img_dir, lbl_dir = image_and_label_dirs
        counts = stratified_split(img_dir, lbl_dir, tmp_path / "split", ratios=(0.7, 0.2, 0.1), seed=42)
        total = counts["train"] + counts["val"] + counts["test"]
        assert total == 10

    def test_output_dirs_exist(self, image_and_label_dirs, tmp_path: Path):
        from src.data.dataset_splitter import stratified_split

        img_dir, lbl_dir = image_and_label_dirs
        out = tmp_path / "split2"
        stratified_split(img_dir, lbl_dir, out, ratios=(0.6, 0.2, 0.2), seed=42)
        for split in ("train", "val", "test"):
            assert (out / split / "images").is_dir()
            assert (out / split / "labels").is_dir()

    def test_ratios_must_sum_to_one(self, image_and_label_dirs, tmp_path: Path):
        from src.data.dataset_splitter import stratified_split

        img_dir, lbl_dir = image_and_label_dirs
        with pytest.raises(AssertionError):
            stratified_split(img_dir, lbl_dir, tmp_path / "bad", ratios=(0.5, 0.1, 0.1))

    def test_compute_class_distribution(self, yolo_label_dir: Path):
        from src.data.dataset_splitter import compute_class_distribution

        dist = compute_class_distribution(yolo_label_dir)
        assert sum(dist.values()) == 10
        assert len(dist) == 5  # classes 0..4


# ─────────────────────────────────────────────────────────────
# Annotation converter
# ─────────────────────────────────────────────────────────────

class TestAnnotationConverter:
    def test_voc_to_yolo(self, tmp_path: Path):
        from src.data.annotation_converter import voc_to_yolo

        voc_xml = tmp_path / "001.xml"
        voc_xml.write_text("""<annotation>
  <size><width>640</width><height>640</height></size>
  <object>
    <name>scratch</name>
    <bndbox><xmin>100</xmin><ymin>200</ymin><xmax>300</xmax><ymax>400</ymax></bndbox>
  </object>
</annotation>""")

        yolo_dir = tmp_path / "yolo"
        yolo_dir.mkdir()
        class_map = {"scratch": 0}
        voc_to_yolo(voc_xml.parent, yolo_dir, class_map)
        txt = (yolo_dir / "001.txt").read_text().strip()
        parts = txt.split()
        assert int(parts[0]) == 0
        assert len(parts) == 5

    def test_coco_to_yolo(self, tmp_path: Path):
        import json
        from src.data.annotation_converter import coco_to_yolo

        coco = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40]}
            ],
            "categories": [{"id": 0, "name": "scratch"}],
        }
        coco_file = tmp_path / "coco.json"
        coco_file.write_text(json.dumps(coco))
        yolo_dir = tmp_path / "yolo_coco"
        yolo_dir.mkdir()
        coco_to_yolo(coco_file, yolo_dir)
        assert (yolo_dir / "a.txt").exists()

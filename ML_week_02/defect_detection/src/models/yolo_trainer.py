"""
yolo_trainer.py – YOLOv8 training wrapper with custom configurations.

Architecture overview:
  Backbone:  CSPDarknet53 + C2f modules
  Neck:      PANet (Path Aggregation Network)
  Head:      Anchor-free detection head  (VFL + DFL + CIoU loss)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.config import load_training_config, merge_configs, project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)


class YOLOv8Trainer:
    """Train, validate, and export YOLOv8 for defect detection."""

    _SIZE_PARAMS = {
        "n": ("3.2 M", 8.7),
        "s": ("11.2 M", 28.6),
        "m": ("25.9 M", 78.9),
        "l": ("43.7 M", 165.2),
        "x": ("68.2 M", 257.8),
    }

    def __init__(
        self,
        model_size: str = "s",
        pretrained: bool = True,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model_size = model_size
        self.device = device

        weight_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
        self.model = YOLO(weight_name)

        params, gflops = self._SIZE_PARAMS.get(model_size, ("?", 0))
        logger.info(f"YOLOv8{model_size.upper()} loaded — {params} params, {gflops} GFLOPs, device={device}")

    # ── Training ─────────────────────────────────────────────

    def train(
        self,
        data_yaml: Path,
        config: Dict[str, Any] | None = None,
        config_name: str = "base",
        project: str | None = None,
        name: str = "defect_detection",
    ) -> Any:
        """Launch a training run.

        Parameters
        ----------
        data_yaml   : path to dataset.yaml
        config      : dict overrides (merged on top of base YAML config)
        config_name : name of the YAML config under configs/training/
        project     : experiment directory
        name        : run name
        """
        base_cfg = load_training_config(config_name)
        if config:
            base_cfg = merge_configs(base_cfg, config)

        train_args: Dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": base_cfg.get("epochs", 100),
            "batch": base_cfg.get("batch", 16),
            "imgsz": base_cfg.get("imgsz", 640),
            "device": self.device,
            "workers": base_cfg.get("workers", 4),
            "project": project or str(project_root() / "experiments" / "runs" / "train"),
            "name": name,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": base_cfg.get("optimizer", "auto"),
            "verbose": base_cfg.get("verbose", True),
            "seed": base_cfg.get("seed", 42),
            "deterministic": True,
            "lr0": base_cfg.get("lr0", 0.01),
            "lrf": base_cfg.get("lrf", 0.01),
            "momentum": base_cfg.get("momentum", 0.937),
            "weight_decay": base_cfg.get("weight_decay", 0.0005),
            "warmup_epochs": base_cfg.get("warmup_epochs", 3.0),
            "box": base_cfg.get("box", 7.5),
            "cls": base_cfg.get("cls", 0.5),
            "dfl": base_cfg.get("dfl", 1.5),
            "patience": base_cfg.get("patience", 50),
            "save": True,
            "plots": True,
            "amp": base_cfg.get("amp", True),
            "close_mosaic": base_cfg.get("close_mosaic", 10),
            "val": True,
        }

        # Augmentation params
        for key in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                     "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste"):
            if key in base_cfg:
                train_args[key] = base_cfg[key]

        logger.info("=" * 60)
        logger.info(f"Training YOLOv8{self.model_size.upper()}")
        logger.info(f"  Dataset  : {data_yaml}")
        logger.info(f"  Epochs   : {train_args['epochs']}")
        logger.info(f"  Batch    : {train_args['batch']}")
        logger.info(f"  ImgSize  : {train_args['imgsz']}")
        logger.info(f"  Device   : {train_args['device']}")
        logger.info("=" * 60)

        results = self.model.train(**train_args)
        logger.success("Training complete!")
        return results

    # ── Validation ───────────────────────────────────────────

    def validate(self, data_yaml: Path, **kwargs) -> Any:
        """Run validation and return metrics (mAP, precision, recall)."""
        results = self.model.val(data=str(data_yaml), device=self.device, **kwargs)
        return results

    # ── Export ────────────────────────────────────────────────

    def export(
        self,
        fmt: str = "onnx",
        simplify: bool = True,
        half: bool = False,
        imgsz: int = 640,
    ) -> Path:
        """Export model to ONNX / CoreML / TorchScript / etc."""
        path = self.model.export(
            format=fmt,
            simplify=simplify if fmt == "onnx" else False,
            half=half,
            imgsz=imgsz,
        )
        logger.success(f"Exported → {path}")
        return Path(path)

    # ── Predict (convenience) ────────────────────────────────

    def predict(self, source, **kwargs):
        """Run inference on *source* (image path, numpy, list, etc.)."""
        return self.model.predict(source, device=self.device, **kwargs)

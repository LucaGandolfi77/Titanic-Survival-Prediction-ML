"""
model_exporter.py – Export trained YOLOv8 weights to deployment formats.

Supports:  ONNX · CoreML · TorchScript · TensorRT · OpenVINO
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

from src.utils.config import project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

EXPORT_DIR = project_root() / "models" / "exported"


def export_model(
    weights_path: Path,
    fmt: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True,
    output_dir: Path | None = None,
) -> Path:
    """Export a .pt checkpoint to the desired format.

    Parameters
    ----------
    weights_path : path to best.pt / last.pt
    fmt          : onnx, coreml, torchscript, engine (TensorRT), openvino
    output_dir   : destination directory (default: models/exported/)

    Returns the path to the exported file.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    export_kwargs = dict(
        format=fmt,
        imgsz=imgsz,
        half=half,
    )
    if fmt == "onnx":
        export_kwargs["simplify"] = simplify

    exported_path = model.export(**export_kwargs)
    exported_path = Path(exported_path)

    dest_dir = Path(output_dir) if output_dir else EXPORT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / exported_path.name
    if exported_path != dest:
        shutil.copy2(exported_path, dest)

    logger.success(f"Exported {weights_path.name} → {dest}  ({fmt})")
    return dest


def list_exports(directory: Path | None = None) -> Dict[str, Path]:
    """List exported model files in *directory*."""
    d = Path(directory) if directory else EXPORT_DIR
    exports: Dict[str, Path] = {}
    for p in sorted(d.iterdir()):
        if p.is_file():
            exports[p.stem] = p
    return exports


def get_model_info(weights_path: Path) -> Dict:
    """Return a summary dict (class count, img size, params, …)."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    n_params = sum(p.numel() for p in model.model.parameters())
    return {
        "weights": str(weights_path),
        "task": model.task,
        "parameters": n_params,
        "names": model.names,
        "nc": len(model.names) if model.names else 0,
    }

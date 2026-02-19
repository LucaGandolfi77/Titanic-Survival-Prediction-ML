"""
downloader.py – Download and prepare public PCB defect datasets.

Supported sources:
  1. DeepPCB (GitHub)      – open-source PCB defect + template pairs
  2. Kaggle PCB Defects    – 1 386 images, 6 classes
  3. Local directory copy   – bring your own images
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetDownloader:
    """Download & extract public defect-detection datasets."""

    def __init__(self, data_dir: Path = Path("data/raw")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Public APIs ──────────────────────────────────────────

    def download_deeppcb(self) -> Path:
        """Download DeepPCB dataset from GitHub.

        Returns path to the extracted folder.
        """
        url = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"
        zip_path = self.data_dir / "deeppcb.zip"

        logger.info("Downloading DeepPCB dataset …")
        self._download_file(url, zip_path)

        logger.info("Extracting …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

        dataset_path = self.data_dir / "DeepPCB-master"
        zip_path.unlink(missing_ok=True)
        logger.success(f"DeepPCB extracted → {dataset_path}")
        return dataset_path

    def download_kaggle_pcb(self) -> Path:
        """Download PCB defects dataset from Kaggle.

        Requires ``~/.kaggle/kaggle.json`` with valid credentials.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError("pip install kaggle — then configure ~/.kaggle/kaggle.json")

        api = KaggleApi()
        api.authenticate()

        out = self.data_dir / "kaggle_pcb"
        logger.info("Downloading kaggle/akhatova/pcb-defects …")
        api.dataset_download_files("akhatova/pcb-defects", path=str(out), unzip=True)
        logger.success(f"Kaggle PCB dataset → {out}")
        return out

    def copy_local(self, source_dir: Path) -> Path:
        """Copy a local image directory into data/raw/."""
        dest = self.data_dir / source_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_dir, dest)
        logger.success(f"Copied local dataset → {dest}")
        return dest

    # ── Internals ────────────────────────────────────────────

    def _download_file(self, url: str, dest: Path) -> None:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            desc=dest.name, total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in resp.iter_content(8192):
                fh.write(chunk)
                bar.update(len(chunk))

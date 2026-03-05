import threading
import logging
import requests
from pathlib import Path
from typing import Callable, Optional
from config import MODEL_CATALOG, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelManager:

    def __init__(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Catalog queries ──────────────────────────────────────────────────────

    def get_catalog(self) -> list[dict]:
        return MODEL_CATALOG

    def get_by_id(self, model_id: str) -> Optional[dict]:
        return next((m for m in MODEL_CATALOG if m["id"] == model_id), None)

    def is_downloaded(self, model_id: str) -> bool:
        m = self.get_by_id(model_id)
        return m is not None and (MODELS_DIR / m["filename"]).exists()

    def get_local_path(self, model_id: str) -> Optional[Path]:
        m = self.get_by_id(model_id)
        if not m:
            return None
        p = MODELS_DIR / m["filename"]
        return p if p.exists() else None

    def list_downloaded(self) -> list[dict]:
        result = []
        for m in MODEL_CATALOG:
            p = MODELS_DIR / m["filename"]
            if p.exists():
                result.append({**m, "path": str(p),
                                "size_bytes": p.stat().st_size})
        return result

    # ── Operations ───────────────────────────────────────────────────────────

    def delete(self, model_id: str) -> bool:
        m = self.get_by_id(model_id)
        if not m:
            return False
        p = MODELS_DIR / m["filename"]
        if p.exists():
            p.unlink()
            logger.info(f"Deleted model file: {p}")
            return True
        return False

    def download(
        self,
        model_id: str,
        on_progress: Callable[[int, int], None] = None,
        on_complete: Callable[[str], None] = None,
        on_error:    Callable[[str], None] = None,
    ) -> Optional[threading.Thread]:

        m = self.get_by_id(model_id)
        if not m:
            if on_error:
                on_error(f"Model '{model_id}' not found in catalog.")
            return None

        def _worker():
            dest = MODELS_DIR / m["filename"]
            try:
                headers = {"User-Agent": "LocalAIRunner/1.0"}
                with requests.get(m["url"], stream=True,
                                  headers=headers, timeout=60) as resp:
                    resp.raise_for_status()
                    total      = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    with open(dest, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=131072):
                            if chunk:
                                fh.write(chunk)
                                downloaded += len(chunk)
                                if on_progress:
                                    on_progress(downloaded, total)

                logger.info(f"Download complete: {dest}")
                if on_complete:
                    on_complete(str(dest))

            except Exception as exc:
                if dest.exists():
                    dest.unlink()
                logger.error(f"Download failed [{model_id}]: {exc}")
                if on_error:
                    on_error(str(exc))

        t = threading.Thread(target=_worker, daemon=True,
                             name=f"dl-{model_id}")
        t.start()
        return t

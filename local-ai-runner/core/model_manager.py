import os
import ssl
import threading
import logging
import requests
import certifi
import urllib3
import json
from pathlib import Path
from typing import Callable, Optional
from config import MODEL_CATALOG, MODELS_DIR

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub import constants as hf_constants
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── SSL helper ────────────────────────────────────────────────────────────────
_CA_BUNDLE = os.environ.get("REQUESTS_CA_BUNDLE", certifi.where())
_VERIFY    = False if str(_CA_BUNDLE).lower() == "false" else _CA_BUNDLE

# Suppress InsecureRequestWarning globally when verify=False
if not _VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Tell huggingface_hub to skip SSL verification for xethub CDN
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")


class ModelManager:

    def __init__(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # index to allow models downloaded to custom locations
        self._index_path = MODELS_DIR / 'models_index.json'
        self._index = {}
        if self._index_path.exists():
            try:
                with open(self._index_path, 'r', encoding='utf8') as fh:
                    self._index = json.load(fh)
            except Exception:
                self._index = {}

    # ── Catalog queries ───────────────────────────────────────────────────────

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
        # check index (custom download locations) first
        if model_id in self._index:
            p = Path(self._index[model_id])
            if p.exists():
                return p
        # fallback to default models dir
        p = MODELS_DIR / m["filename"]
        return p if p.exists() else None

    def list_downloaded(self) -> list[dict]:
        result = []
        # include indexed locations
        for mid, pstr in self._index.items():
            p = Path(pstr)
            m = self.get_by_id(mid) or {"id": mid, "name": mid, "filename": p.name}
            if p.exists():
                result.append({**m, "path": str(p), "size_bytes": p.stat().st_size})
        # include any files in MODELS_DIR
        for m in MODEL_CATALOG:
            p = MODELS_DIR / m["filename"]
            if p.exists():
                result.append({**m, "path": str(p), "size_bytes": p.stat().st_size})
        return result

    # ── Operations ────────────────────────────────────────────────────────────

    def delete(self, model_id: str) -> bool:
        m = self.get_by_id(model_id)
        if not m:
            return False
        # if indexed, remove that path
        if model_id in self._index:
            try:
                p = Path(self._index[model_id])
                if p.exists():
                    p.unlink()
                del self._index[model_id]
                self._save_index()
                logger.info(f"Deleted model file (indexed): {p}")
                return True
            except Exception:
                return False
        # else remove from MODELS_DIR
        p = MODELS_DIR / m["filename"]
        if p.exists():
            p.unlink()
            logger.info(f"Deleted model file: {p}")
            return True
        return False

    def _save_index(self):
        try:
            with open(self._index_path, 'w', encoding='utf8') as fh:
                json.dump(self._index, fh, indent=2)
        except Exception:
            pass

    def register_local_path(self, model_id: str, path: str | Path) -> bool:
        """Register an already-downloaded local model file for a catalog entry.

        This records the absolute path in the index so the UI can load it later.
        Returns True on success, False otherwise.
        """
        m = self.get_by_id(model_id)
        if not m:
            return False
        p = Path(path)
        if not p.exists():
            return False
        try:
            self._index[model_id] = str(p.resolve())
            self._save_index()
            logger.info(f"Registered local model: {model_id} -> {p}")
            return True
        except Exception:
            return False

    def download(
        self,
        model_id: str,
        dest_dir: Optional[str | Path] = None,
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
            dest_dir_path = Path(dest_dir) if dest_dir else MODELS_DIR
            dest_dir_path.mkdir(parents=True, exist_ok=True)
            dest = dest_dir_path / m["filename"]
            success = self._hf_download(m, dest, on_progress,
                                        on_complete, on_error)
            if not success:
                logger.info(f"[{model_id}] Falling back to requests download.")
                self._requests_download(m, dest, on_progress,
                                        on_complete, on_error)
            # register in index if destination is outside MODELS_DIR
            try:
                if dest.exists():
                    if dest_dir and Path(dest_dir).resolve() != MODELS_DIR.resolve():
                        self._index[model_id] = str(dest.resolve())
                        self._save_index()
                    if on_complete:
                        on_complete(str(dest))
            except Exception:
                pass

        t = threading.Thread(target=_worker, daemon=True, name=f"dl-{model_id}")
        t.start()
        return t

    # ── Private: HuggingFace Hub download ─────────────────────────────────────

    def _hf_download(
        self,
        m: dict,
        dest: Path,
        on_progress: Callable[[int, int], None],
        on_complete: Callable[[str], None],
        on_error:    Callable[[str], None],
    ) -> bool:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub import constants as hf_constants
        except ImportError:
            logger.warning("huggingface_hub not installed. "
                           "Run: pip install huggingface_hub")
            return False

        try:
            # Force SSL verification off for the entire hf_hub session
            # This covers both hf.co and cas-bridge.xethub.hf.co
            hf_constants.HF_HUB_DISABLE_SSL_VERIFICATION = True
            os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

            # Resolve repo_id and filename
            if "repo_id" in m:
                repo_id     = m["repo_id"]
                hf_filename = m["filename"]
            else:
                repo_id, hf_filename = self._parse_hf_url(m["url"])

            total_bytes = int(m.get("size_gb", 1) * 1024 ** 3)

            # Progress polling thread
            stop_poll   = threading.Event()
            poll_thread = None
            if on_progress:
                def _poll():
                    while not stop_poll.is_set():
                        if dest.exists():
                            try:
                                on_progress(dest.stat().st_size, total_bytes)
                            except Exception:
                                pass
                        stop_poll.wait(1.5)

                poll_thread = threading.Thread(target=_poll, daemon=True,
                                               name="hf-progress-poll")
                poll_thread.start()

            logger.info(f"[HF] Downloading {repo_id}/{hf_filename} → {dest}")
            # Use HF token if provided in environment to access gated repos
            hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if hf_token:
                logger.info("Using HUGGINGFACE_HUB_TOKEN for authenticated hf_hub_download.")

            cached = hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
                token=hf_token,
            )

            stop_poll.set()
            if poll_thread:
                poll_thread.join(timeout=2)

            cached_path = Path(cached).resolve()
            if cached_path != dest.resolve() and cached_path.exists():
                cached_path.rename(dest)
                logger.info(f"[HF] Moved {cached_path} → {dest}")

            if on_progress:
                on_progress(dest.stat().st_size, dest.stat().st_size)

            logger.info(f"[HF] Download complete: {dest}")
            if on_complete:
                on_complete(str(dest))
            return True

        except Exception as exc:
            logger.warning(f"[HF] Download failed: {exc} — trying fallback.")
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            return False

    # ── Private: requests fallback download ───────────────────────────────────

    def _requests_download(
        self,
        m: dict,
        dest: Path,
        on_progress: Callable[[int, int], None],
        on_complete: Callable[[str], None],
        on_error:    Callable[[str], None],
    ) -> None:

        try:
            session = requests.Session()
            session.verify = False      # skip SSL — same fix as hf_hub above
            session.headers.update({"User-Agent": "LocalAIRunner/1.0"})

            # If the user has set a HuggingFace token, use it for requests fallback too
            http_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if http_token:
                session.headers.update({"Authorization": f"Bearer {http_token}"})

            logger.info(f"[requests] Downloading {m['url']} → {dest}")
            with session.get(m["url"], stream=True, timeout=60) as resp:
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

            logger.info(f"[requests] Download complete: {dest}")
            if on_complete:
                on_complete(str(dest))

        except Exception as exc:
            if dest.exists():
                dest.unlink()
            logger.error(f"[requests] Download failed [{m['id']}]: {exc}")
            if on_error:
                on_error(str(exc))

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_hf_url(url: str) -> tuple[str, str]:
        path  = url.replace("https://huggingface.co/", "")
        parts = path.split("/")
        if len(parts) < 5 or parts[2] != "resolve":
            raise ValueError(f"Cannot parse HuggingFace URL: {url}")
        repo_id  = f"{parts[0]}/{parts[1]}"
        filename = "/".join(parts[4:])
        return repo_id, filename

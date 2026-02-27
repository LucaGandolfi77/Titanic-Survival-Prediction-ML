"""Asynchronous image downloader using *aiohttp*.

Downloads product images concurrently (with a semaphore to cap
parallelism) and writes them to a local folder.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import aiohttp

from utils.logger import get_logger
from utils.validator import safe_filename

log = get_logger(__name__)


class ImageDownloader:
    """Download product images concurrently.

    Args:
        output_folder: Directory to save downloaded images.
        max_concurrent: Maximum simultaneous downloads.
    """

    def __init__(
        self,
        output_folder: str | Path = "./output/images",
        max_concurrent: int = 5,
    ) -> None:
        self._output = Path(output_folder).resolve()
        self._output.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent

    # ── Single image ────────────────────────────────────────────

    async def download_image(
        self,
        url: str,
        filename: str,
        session: aiohttp.ClientSession | None = None,
    ) -> str | None:
        """Download one image and save it locally.

        Args:
            url: Absolute URL of the image.
            filename: Desired stem (without extension); the extension is
                derived from the response content-type.
            session: Shared aiohttp session (created internally if not
                provided).

        Returns:
            Absolute local file path on success, or ``None``.
        """
        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()

        try:
            async with self._semaphore:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        log.warning("Image download HTTP %d: %s", resp.status, url)
                        return None

                    content_type = resp.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        log.warning("Non-image content-type '%s' for %s", content_type, url)
                        return None

                    ext = self._extension_from_ct(content_type, url)
                    safe_name = safe_filename(filename)
                    dest = self._output / f"{safe_name}{ext}"

                    # Avoid name collisions
                    counter = 1
                    while dest.exists():
                        dest = self._output / f"{safe_name}_{counter}{ext}"
                        counter += 1

                    data = await resp.read()
                    dest.write_bytes(data)
                    log.debug("Saved image → %s", dest)
                    return str(dest)

        except asyncio.TimeoutError:
            log.warning("Timeout downloading image: %s", url)
            return None
        except aiohttp.ClientError as exc:
            log.warning("aiohttp error for %s: %s", url, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected error downloading %s: %s", url, exc)
            return None
        finally:
            if own_session and session:
                await session.close()

    # ── Batch download ──────────────────────────────────────────

    async def download_all(
        self,
        products: list[dict[str, Any]],
        progress_callback: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Download images for all products concurrently.

        Each product dict must contain ``image_url`` and ``name`` keys.
        A ``local_image_path`` key is added to every product after
        processing.

        Args:
            products: Product dicts (modified in place).
            progress_callback: Optional callable ``(done, total)`` called
                after each image completes.

        Returns:
            The same *products* list with ``local_image_path`` populated.
        """
        total = len(products)
        done = 0

        async with aiohttp.ClientSession() as session:
            tasks: list[asyncio.Task[str | None]] = []
            for idx, prod in enumerate(products):
                url = prod.get("image_url", "")
                if not url or url == "N/A":
                    prod["local_image_path"] = ""
                    done += 1
                    if progress_callback:
                        progress_callback(done, total)
                    continue

                name = prod.get("name", f"product_{idx}")
                task = asyncio.create_task(
                    self._download_and_update(session, prod, url, name)
                )
                tasks.append(task)

            for coro in asyncio.as_completed(tasks):
                await coro
                done += 1
                if progress_callback:
                    progress_callback(done, total)

        # Ensure all products have the key
        for prod in products:
            prod.setdefault("local_image_path", "")

        images_ok = sum(1 for p in products if p.get("local_image_path"))
        log.info("Downloaded %d / %d images", images_ok, total)
        return products

    async def _download_and_update(
        self,
        session: aiohttp.ClientSession,
        product: dict[str, Any],
        url: str,
        name: str,
    ) -> None:
        """Download a single image and update the product dict in place."""
        path = await self.download_image(url, name, session)
        product["local_image_path"] = path or ""

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extension_from_ct(content_type: str, url: str) -> str:
        """Derive file extension from Content-Type or URL.

        Args:
            content_type: MIME type string.
            url: Image URL (fallback for extension).

        Returns:
            Extension string including the dot, e.g. ``".jpg"``.
        """
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            # mimetypes returns ".jpe" for jpeg sometimes – normalise
            if ext in (".jpe", ".jpeg"):
                ext = ".jpg"
            return ext

        # Fallback: try URL path
        match = re.search(r"\.(jpe?g|png|webp|gif|svg|bmp|tiff?)(\?|$)", url, re.I)
        if match:
            return f".{match.group(1).lower()}"

        return ".jpg"  # ultimate fallback


def run_download(
    products: list[dict[str, Any]],
    output_folder: str | Path = "./output/images",
    max_concurrent: int = 5,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """Convenience synchronous wrapper around :meth:`ImageDownloader.download_all`.

    Args:
        products: Product dicts to enrich with local image paths.
        output_folder: Where to save images.
        max_concurrent: Semaphore limit.
        progress_callback: Optional ``(done, total)`` callable.

    Returns:
        Updated products list.
    """
    downloader = ImageDownloader(output_folder, max_concurrent)

    # Use existing event loop or create one
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We are inside an already-running loop (e.g. Jupyter / Tkinter)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            result = pool.submit(
                asyncio.run,
                downloader.download_all(products, progress_callback),
            ).result()
        return result  # type: ignore[return-value]
    else:
        return asyncio.run(
            downloader.download_all(products, progress_callback)
        )

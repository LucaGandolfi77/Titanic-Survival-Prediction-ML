"""Fallback CSV exporter for product data.

Writes a simple UTF-8 CSV using :mod:`csv` (no Pandas dependency
required).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

_FIELDNAMES = [
    "index",
    "name",
    "price",
    "description",
    "product_url",
    "image_url",
    "local_image_path",
    "category",
]


class CsvExporter:
    """Export scraped product data to a CSV file.

    Args:
        output_path: Destination file path for the CSV.
    """

    def __init__(self, output_path: str | Path) -> None:
        self._output_path = Path(output_path).resolve()
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, products: list[dict[str, Any]]) -> str:
        """Write *products* to a CSV file.

        Args:
            products: List of product dicts.

        Returns:
            Absolute path to the saved CSV file.
        """
        with self._output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=_FIELDNAMES,
                extrasaction="ignore",
            )
            writer.writeheader()
            for idx, prod in enumerate(products, start=1):
                row = {
                    "index": idx,
                    "name": prod.get("name", "N/A"),
                    "price": prod.get("price", "N/A"),
                    "description": prod.get("description", "N/A"),
                    "product_url": prod.get("product_url", "N/A"),
                    "image_url": prod.get("image_url", "N/A"),
                    "local_image_path": prod.get("local_image_path", ""),
                    "category": prod.get("category", ""),
                }
                writer.writerow(row)

        log.info("CSV exported → %s (%d products)", self._output_path, len(products))
        return str(self._output_path)

"""CLI entry point for WebScraper Pro.

Usage::

    python main.py --url "https://example.com/shop" --max-pages 10
    python main.py --url "https://example.com/shop" --selector-preset shopify
    python main.py --url "https://example.com/shop" --no-images --output ./results

Run ``python main.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

# ── Ensure project root is on sys.path ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.image_downloader import run_download
from core.scraper import ProductScraper
from exporters.csv_exporter import CsvExporter
from exporters.excel_exporter import ExcelExporter
from utils.logger import get_logger
from utils.validator import is_valid_url, normalise_url

log = get_logger(__name__)

# ── Built-in selector presets ───────────────────────────────────

_PRESETS: dict[str, dict[str, str]] = {}


def _load_presets(config_path: Path) -> None:
    """Populate ``_PRESETS`` from the config file's ``presets`` section."""
    global _PRESETS  # noqa: PLW0603
    if _PRESETS:
        return
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        _PRESETS.update(cfg.get("presets", {}))
    # Fallbacks if config is missing
    _PRESETS.setdefault("woocommerce", {
        "product_container": "li.product",
        "name": ".woocommerce-loop-product__title",
        "price": "span.price",
        "description": ".woocommerce-product-details__short-description",
        "image": "img.attachment-woocommerce_thumbnail",
        "next_page": "a.next",
        "pagination_type": "query_param",
        "page_param": "paged",
    })
    _PRESETS.setdefault("shopify", {
        "product_container": "div.product-card",
        "name": ".product-card__title",
        "price": "span.price-item",
        "description": ".product-card__description",
        "image": "img.product-featured-media",
        "next_page": "a[rel='next']",
        "pagination_type": "query_param",
        "page_param": "page",
    })
    _PRESETS.setdefault("generic", {
        "product_container": "div.product, li.product, div.product-item, article.product",
        "name": "h2.product-title, h3.product-title, .product-name, .product-title",
        "price": "span.price, div.price, .product-price, .amount",
        "description": "p.product-description, .product-desc, .short-description",
        "image": "img.product-image, img.product-img, .product-thumbnail img",
        "next_page": "a.next, a.next-page, a[rel='next'], .pagination .next a",
        "pagination_type": "query_param",
        "page_param": "page",
    })


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="webscraper-pro",
        description="🕷️  WebScraper Pro — E-commerce Data Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --url https://example.com/shop\n"
            "  python main.py --url https://example.com/shop --selector-preset shopify\n"
            "  python main.py --url https://shop.example.com --max-pages 5 --no-images\n"
        ),
    )
    parser.add_argument(
        "--url", required=True,
        help="Target URL of the product listing page",
    )
    parser.add_argument(
        "--max-pages", type=int, default=50,
        help="Maximum number of pages to scrape (default: 50)",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Skip image downloading",
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output folder for Excel/CSV and images (default: ./output)",
    )
    parser.add_argument(
        "--selector-preset", type=str, default=None,
        choices=["woocommerce", "shopify", "generic"],
        help="Use a built-in CSS-selector preset instead of config.yaml selectors",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI scraping pipeline.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate URL
    url = normalise_url(args.url)
    if not is_valid_url(url):
        print(f"❌  Invalid URL: {args.url}")
        return 1

    config_path = Path(args.config).resolve()
    _load_presets(config_path)

    # Selector overrides from preset
    selector_overrides: dict[str, str] | None = None
    if args.selector_preset:
        selector_overrides = _PRESETS.get(args.selector_preset)
        if selector_overrides:
            print(f"📋  Using selector preset: {args.selector_preset}")
        else:
            print(f"⚠️  Unknown preset '{args.selector_preset}' — using config defaults")

    output_folder = Path(args.output).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    scraper = ProductScraper(
        config_path=str(config_path),
        selector_overrides=selector_overrides,
        stop_event=stop_event,
    )
    scraper._max_pages = args.max_pages

    # ── Progress bar ────────────────────────────────────────────
    pbar = tqdm(total=args.max_pages, desc="Pages", unit="pg", dynamic_ncols=True)

    def on_page(page: int, items: int) -> None:
        pbar.update(1)
        pbar.set_postfix(products=items)

    print(f"\n🕷️  Scraping: {url}")
    print(f"   Max pages : {args.max_pages}")
    print(f"   Output    : {output_folder}\n")

    products = scraper.scrape_all_pages(url, callback=on_page)
    pbar.close()

    if not products:
        print("\n⚠️  No products found.")
        scraper.close()
        return 0

    print(f"\n✅  Found {len(products)} products across {scraper.pages_visited} pages")

    # ── Download images ─────────────────────────────────────────
    if not args.no_images:
        images_folder = output_folder / "images"
        print(f"📥  Downloading images to {images_folder} …")

        img_pbar = tqdm(total=len(products), desc="Images", unit="img", dynamic_ncols=True)

        def img_progress(done: int, total: int) -> None:
            img_pbar.update(1)

        products = run_download(
            products,
            output_folder=str(images_folder),
            max_concurrent=5,
            progress_callback=img_progress,
        )
        img_pbar.close()

        images_ok = sum(1 for p in products if p.get("local_image_path"))
        print(f"   Images downloaded: {images_ok}/{len(products)}")

    # ── Export Excel ─────────────────────────────────────────────
    excel_path = output_folder / "products.xlsx"
    excel_exporter = ExcelExporter(excel_path)
    saved = excel_exporter.export(
        products,
        source_url=url,
        pages_visited=scraper.pages_visited,
    )
    print(f"📊  Excel saved: {saved}")

    # ── Export CSV ───────────────────────────────────────────────
    csv_path = output_folder / "products.csv"
    csv_saved = CsvExporter(csv_path).export(products)
    print(f"📄  CSV saved  : {csv_saved}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  SCRAPING SUMMARY")
    print("═" * 50)
    print(f"  Source URL      : {url}")
    print(f"  Pages visited   : {scraper.pages_visited}")
    print(f"  Products found  : {len(products)}")
    if not args.no_images:
        images_ok = sum(1 for p in products if p.get("local_image_path"))
        print(f"  Images saved    : {images_ok}/{len(products)}")
    print(f"  Excel file      : {saved}")
    print(f"  CSV file        : {csv_saved}")
    print("═" * 50 + "\n")

    scraper.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""HTTP fetcher + HTML parser for e-commerce product pages.

Provides :class:`ProductScraper` which:

* Fetches pages with retry / exponential back-off
* Rotates User-Agents via :class:`core.anti_bot.AntiBot`
* Respects ``robots.txt`` via :mod:`urllib.robotparser`
* Parses products with configurable CSS selectors
* Handles pagination through :class:`core.paginator.Paginator`

.. note::

   For JavaScript-rendered pages (React / Vue SPAs) swap
   ``requests.get`` for Playwright::

       # pip install playwright && python -m playwright install
       from playwright.sync_api import sync_playwright

       def fetch_page_js(url: str) -> str:
           with sync_playwright() as p:
               browser = p.chromium.launch(headless=True)
               page = browser.new_page()
               page.goto(url, wait_until="networkidle")
               html = page.content()
               browser.close()
               return html

   Then pass the returned HTML into ``BeautifulSoup(html, "lxml")``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import yaml
from bs4 import BeautifulSoup, Tag

from core.anti_bot import AntiBot
from core.browser_fetcher import BrowserFetcher, is_playwright_available
from core.paginator import Paginator
from core.tls_fetcher import TLSFetcher, is_curl_cffi_available
from utils.logger import get_logger
from utils.validator import detect_captcha, sanitise_price, sanitise_text

log = get_logger(__name__)


class ProductScraper:
    """Configurable product-page scraper.

    Args:
        config_path: Path to the YAML configuration file.
        selector_overrides: Dict that overrides individual selectors from
            the config file (used by presets).
        stop_event: Optional :class:`threading.Event` checked between pages
            to allow graceful cancellation from the GUI.
    """

    def __init__(
        self,
        config_path: str | Path = "config/config.yaml",
        selector_overrides: dict[str, str] | None = None,
        stop_event: threading.Event | None = None,
        use_browser: bool = False,
    ) -> None:
        self._config = self._load_config(Path(config_path))
        self._stop_event = stop_event or threading.Event()

        # Merge selector overrides
        selectors: dict[str, str] = dict(self._config.get("selectors", {}))
        if selector_overrides:
            selectors.update(selector_overrides)
        self._selectors = selectors

        scraper_cfg = self._config.get("scraper", {})
        self._timeout: int = scraper_cfg.get("timeout", 15)
        self._max_retries: int = scraper_cfg.get("max_retries", 3)
        self._max_pages: int = scraper_cfg.get("max_pages", 50)

        self._anti_bot = AntiBot(
            user_agents=scraper_cfg.get("user_agents", [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ]),
            delay_min=scraper_cfg.get("delay_min", 1.5),
            delay_max=scraper_cfg.get("delay_max", 3.5),
        )

        self._paginator = Paginator(
            next_page_selector=selectors.get("next_page", "a.next-page"),
            pagination_type=selectors.get("pagination_type", "query_param"),
            page_param=selectors.get("page_param", "page"),
        )

        self._session = requests.Session()
        self._robots_cache: dict[str, RobotFileParser] = {}
        self._lock = threading.Lock()

        # Browser-based fetching (for WAF-protected / JS-rendered sites)
        self._use_browser = use_browser
        self._browser_fetcher: BrowserFetcher | None = None
        if use_browser:
            if is_playwright_available():
                self._browser_fetcher = BrowserFetcher()
                log.info("Browser mode enabled (Playwright)")
            else:
                log.warning(
                    "Browser mode requested but Playwright is not installed. "
                    "Run: pip install playwright && python -m playwright install chromium"
                )
                self._use_browser = False

        # Statistics
        self.pages_visited: int = 0
        self.total_products: int = 0

    # ── Configuration ───────────────────────────────────────────

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        """Load and return the YAML config file.

        Args:
            path: Path to configuration file.

        Returns:
            Parsed config as a dict.
        """
        path = path.resolve()
        if not path.exists():
            log.warning("Config file not found at %s – using defaults", path)
            return {}
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        log.info("Loaded config from %s", path)
        return cfg

    @property
    def config(self) -> dict[str, Any]:
        """Return the raw config dict (read-only copy)."""
        return dict(self._config)

    # ── robots.txt ──────────────────────────────────────────────

    def _is_allowed(self, url: str) -> bool:
        """Check whether the URL is allowed by robots.txt.

        Args:
            url: Absolute URL to check.

        Returns:
            ``True`` if we may fetch the URL.
        """
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        with self._lock:
            if origin not in self._robots_cache:
                rp = RobotFileParser()
                robots_url = f"{origin}/robots.txt"
                rp.set_url(robots_url)
                try:
                    # Fetch robots.txt manually with a timeout to avoid
                    # hanging indefinitely (urllib.robotparser.read() has
                    # no timeout parameter).
                    resp = self._session.get(
                        robots_url,
                        timeout=self._timeout,
                        headers=self._anti_bot.get_headers(),
                    )
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                    else:
                        # No valid robots.txt → allow everything
                        rp.allow_all = True  # type: ignore[attr-defined]
                    log.info("Loaded robots.txt from %s (HTTP %d)",
                             robots_url, resp.status_code)
                except Exception:  # noqa: BLE001
                    log.warning("Could not fetch robots.txt for %s – allowing access", origin)
                    rp.allow_all = True  # type: ignore[attr-defined]
                self._robots_cache[origin] = rp

            rp = self._robots_cache[origin]

        try:
            allowed = rp.can_fetch("*", url)
        except Exception:  # noqa: BLE001
            allowed = True

        if not allowed:
            log.warning("robots.txt disallows: %s", url)
        return allowed

    # ── Fetch ───────────────────────────────────────────────────

    def fetch_page(self, url: str) -> BeautifulSoup | None:
        """Fetch a single page and return parsed HTML.

        Implements retry with exponential back-off, User-Agent rotation, and
        random delay between requests.

        Args:
            url: Absolute URL to fetch.

        Returns:
            A :class:`BeautifulSoup` tree, or ``None`` on failure.
        """
        if not self._is_allowed(url):
            log.warning("Skipping disallowed URL: %s", url)
            return None

        last_error: str = ""
        for attempt in range(1, self._max_retries + 1):
            if self._stop_event.is_set():
                log.info("Stop signal received – aborting fetch")
                return None

            headers = self._anti_bot.get_headers()
            try:
                log.debug("GET %s (attempt %d/%d)", url, attempt, self._max_retries)
                resp = self._session.get(url, headers=headers, timeout=self._timeout)

                if resp.status_code == 429:
                    wait = min(2 ** attempt * 2, 30)
                    log.warning("429 Rate Limited – waiting %d s", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code == 403:
                    log.warning("403 Forbidden for %s", url)

                    # ── Fallback chain: TLS impersonation → headless browser ──

                    # 1) Try curl_cffi TLS impersonation (fast, no browser needed)
                    result = self._fetch_with_tls(url)
                    if result is not None:
                        return result

                    # 2) Headless browser fallback
                    if self._browser_fetcher:
                        log.info("Retrying with headless browser …")
                        return self._fetch_with_browser(url)
                    if is_playwright_available() and not self._browser_fetcher:
                        log.info("Auto-enabling browser mode after 403")
                        self._browser_fetcher = BrowserFetcher()
                        self._use_browser = True
                        return self._fetch_with_browser(url)

                    log.error(
                        "403 Forbidden and no fallback backend available. "
                        "Install curl_cffi or Playwright."
                    )
                    return None

                if resp.status_code == 404:
                    log.error("404 Not Found: %s", url)
                    return None

                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    log.warning("%d Server Error – retrying in %d s", resp.status_code, wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()

                # CAPTCHA / access-denied detection
                if detect_captcha(resp.text):
                    log.warning("CAPTCHA detected on %s – skipping page", url)
                    return None

                self._anti_bot.random_delay()
                return BeautifulSoup(resp.text, "lxml")

            except requests.RequestException as exc:
                last_error = str(exc)
                wait = 2 ** attempt
                log.warning("Request failed (attempt %d): %s – retrying in %d s",
                            attempt, last_error, wait)
                time.sleep(wait)

        log.error("All %d retries exhausted for %s: %s",
                  self._max_retries, url, last_error)
        return None

    # ── Parse ───────────────────────────────────────────────────

    def parse_products(
        self, soup: BeautifulSoup, page_url: str = ""
    ) -> list[dict[str, str]]:
        """Extract product data from a parsed page.

        Args:
            soup: Parsed HTML of a product listing page.
            page_url: Used to resolve relative image/product URLs.

        Returns:
            List of product dicts with keys: ``name``, ``price``,
            ``description``, ``image_url``, ``product_url``.
        """
        container_sel = self._selectors.get("product_container", "div.product-item")
        containers: list[Tag] = soup.select(container_sel)

        if not containers:
            log.warning("No product containers found with selector '%s'", container_sel)
            return []

        products: list[dict[str, str]] = []
        for tag in containers:
            product = self._extract_one(tag, page_url)
            products.append(product)

        log.info("Parsed %d products from page", len(products))
        return products

    def _extract_one(self, tag: Tag, page_url: str) -> dict[str, str]:
        """Extract a single product dict from a container tag.

        Args:
            tag: The product container element.
            page_url: Base URL for resolving relative hrefs.

        Returns:
            Dict with product fields.
        """
        name = self._text(tag, self._selectors.get("name", "h2.product-title"))
        price = sanitise_price(
            self._raw_text(tag, self._selectors.get("price", "span.price"))
        )
        description = self._text(
            tag, self._selectors.get("description", "p.product-description")
        )

        # Image URL
        image_url = "N/A"
        img_sel = self._selectors.get("image", "img.product-image")
        img_tag = tag.select_one(img_sel)
        if img_tag:
            image_url = (
                img_tag.get("src")
                or img_tag.get("data-src")
                or img_tag.get("data-lazy-src")
                or "N/A"
            )
            if isinstance(image_url, list):
                image_url = image_url[0] if image_url else "N/A"
            if image_url != "N/A" and not image_url.startswith(("http://", "https://")):
                image_url = urljoin(page_url, image_url)

        # Product link
        product_url = "N/A"
        link_tag = tag.select_one("a[href]")
        if link_tag:
            href = link_tag.get("href", "")
            if isinstance(href, list):
                href = href[0] if href else ""
            if href and href != "#":
                product_url = href if href.startswith("http") else urljoin(page_url, href)

        return {
            "name": name,
            "price": price,
            "description": description,
            "image_url": image_url,
            "product_url": product_url,
        }

    # ── Pagination ──────────────────────────────────────────────

    def get_next_page_url(
        self, soup: BeautifulSoup, current_url: str
    ) -> str | None:
        """Determine the URL of the next listing page.

        Args:
            soup: Parsed HTML of the current page.
            current_url: The URL of the current page.

        Returns:
            Next page URL or ``None``.
        """
        return self._paginator.find_next_url(soup, current_url)

    # ── Full scrape ─────────────────────────────────────────────

    def scrape_all_pages(
        self,
        base_url: str,
        callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, str]]:
        """Scrape all pages starting from *base_url*.

        Args:
            base_url: First page of the product listing.
            callback: Optional function called as
                ``callback(current_page, total_items_so_far)`` after each
                page for progress reporting (e.g. GUI updates).

        Returns:
            Aggregated list of product dicts from all pages.
        """
        all_products: list[dict[str, str]] = []
        current_url: str | None = base_url
        page_num = 0

        while current_url and page_num < self._max_pages:
            if self._stop_event.is_set():
                log.info("Stop signal received – ending pagination")
                break

            page_num += 1
            log.info("── Page %d: %s", page_num, current_url)

            soup = self.fetch_page(current_url)
            if soup is None:
                log.warning("Failed to fetch page %d – stopping", page_num)
                break

            products = self.parse_products(soup, current_url)
            all_products.extend(products)

            self.pages_visited = page_num
            self.total_products = len(all_products)

            if callback:
                callback(page_num, len(all_products))

            current_url = self.get_next_page_url(soup, current_url)

        log.info(
            "Scraping complete: %d products across %d pages",
            len(all_products), page_num,
        )
        return all_products

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _text(parent: Tag, selector: str) -> str:
        el = parent.select_one(selector)
        return sanitise_text(el.get_text() if el else None)

    @staticmethod
    def _raw_text(parent: Tag, selector: str) -> str | None:
        el = parent.select_one(selector)
        return el.get_text() if el else None

    def _fetch_with_tls(self, url: str) -> BeautifulSoup | None:
        """Try fetching with TLS-impersonation (``curl_cffi``).

        Returns parsed HTML on success, ``None`` on failure.
        """
        if not is_curl_cffi_available():
            return None
        try:
            fetcher = TLSFetcher(timeout=self._timeout)
            log.info("Retrying with TLS impersonation (curl_cffi) …")
            html = fetcher.fetch(url)
            fetcher.close()
            if not html:
                return None
            if detect_captcha(html):
                log.warning("CAPTCHA detected (TLS) on %s – skipping", url)
                return None
            self._anti_bot.random_delay()
            return BeautifulSoup(html, "lxml")
        except Exception as exc:  # noqa: BLE001
            log.debug("TLS fallback failed: %s", exc)
            return None

    def _fetch_with_browser(self, url: str) -> BeautifulSoup | None:
        """Fetch a page using the headless browser fallback.

        Args:
            url: Absolute URL to fetch.

        Returns:
            Parsed HTML or ``None``.
        """
        if not self._browser_fetcher:
            return None
        html = self._browser_fetcher.fetch(url)
        if not html:
            return None
        if detect_captcha(html):
            log.warning("CAPTCHA detected (browser) on %s – skipping", url)
            return None
        self._anti_bot.random_delay()
        return BeautifulSoup(html, "lxml")

    def close(self) -> None:
        """Close the underlying HTTP session and browser."""
        self._session.close()
        if self._browser_fetcher:
            self._browser_fetcher.close()
            self._browser_fetcher = None
        log.debug("HTTP session closed")

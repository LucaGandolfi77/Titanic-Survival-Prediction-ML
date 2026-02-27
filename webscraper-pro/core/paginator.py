"""Pagination helpers for multi-page product listings.

Supports three strategies:

* **query_param** – ``?page=2``, ``?paged=3``
* **path** – ``/page/2/``
* **button** – detect a "Load More" button selector
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from utils.logger import get_logger

log = get_logger(__name__)


class Paginator:
    """Build and discover pagination URLs.

    Args:
        next_page_selector: CSS selector for the "next page" link.
        pagination_type: One of ``query_param``, ``path``, ``button``.
        page_param: Query-string parameter name (only used with
            ``query_param``).
    """

    VALID_TYPES = {"query_param", "path", "button"}

    def __init__(
        self,
        next_page_selector: str = "a.next-page",
        pagination_type: str = "query_param",
        page_param: str = "page",
    ) -> None:
        self._selector = next_page_selector
        self._type = pagination_type if pagination_type in self.VALID_TYPES else "query_param"
        self._param = page_param

    # ── Public API ──────────────────────────────────────────────

    def detect_pagination_type(self, soup: BeautifulSoup) -> str:
        """Attempt to auto-detect the pagination strategy from the HTML.

        Falls back to the configured type if detection is inconclusive.

        Args:
            soup: Parsed HTML page.

        Returns:
            One of ``query_param``, ``path``, ``button``.
        """
        next_link = soup.select_one(self._selector)
        if next_link:
            href = next_link.get("href", "")
            if isinstance(href, list):
                href = href[0] if href else ""
            if re.search(r"[?&]page=", href, re.IGNORECASE):
                return "query_param"
            if re.search(r"/page/\d+", href):
                return "path"
        # Check for "load more" button patterns
        load_more = soup.select_one(
            "button.load-more, a.load-more, [data-action='load-more']"
        )
        if load_more:
            return "button"
        return self._type

    def build_page_url(
        self, base_url: str, page_num: int, pagination_type: str | None = None
    ) -> str:
        """Construct the URL for a given page number.

        Args:
            base_url: The original listing URL.
            page_num: Target page number (1-based).
            pagination_type: Override the instance pagination type.

        Returns:
            Fully-qualified URL for the requested page.
        """
        ptype = pagination_type or self._type

        if ptype == "path":
            # Remove existing /page/N/ segment, then append new one
            clean = re.sub(r"/page/\d+/?", "/", base_url).rstrip("/")
            return f"{clean}/page/{page_num}/"

        # Default: query_param
        parsed = urlparse(base_url)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        qs[self._param] = [str(page_num)]
        new_query = urlencode(qs, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    def find_next_url(
        self, soup: BeautifulSoup, current_url: str = ""
    ) -> str | None:
        """Find the next-page URL from the parsed HTML.

        Args:
            soup: Parsed HTML of the current page.
            current_url: Used to resolve relative links.

        Returns:
            Absolute URL of the next page, or ``None`` if there is none.
        """
        link = soup.select_one(self._selector)
        if not link:
            log.debug("No next-page element found for selector '%s'", self._selector)
            return None

        href = link.get("href")
        if not href:
            log.debug("Next-page element has no href attribute")
            return None

        if isinstance(href, list):
            href = href[0] if href else ""

        href = href.strip()
        if not href or href == "#":
            return None

        # Resolve relative URLs
        if not href.startswith(("http://", "https://")):
            href = urljoin(current_url, href)

        log.debug("Next-page URL: %s", href)
        return href

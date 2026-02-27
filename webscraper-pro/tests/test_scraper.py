"""Unit tests for the scraping pipeline.

Tests use synthetic HTML fixtures — no network access required.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

# ── Ensure project root is importable ───────────────────────────
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.anti_bot import AntiBot
from core.paginator import Paginator
from core.scraper import ProductScraper
from utils.validator import (
    detect_captcha,
    is_valid_url,
    normalise_url,
    safe_filename,
    sanitise_price,
    sanitise_text,
)


# ═══════════════════════════════════════════════════════════════
# Validator tests
# ═══════════════════════════════════════════════════════════════

class TestIsValidUrl:
    """Tests for ``is_valid_url``."""

    def test_valid_https(self) -> None:
        assert is_valid_url("https://example.com/shop") is True

    def test_valid_http(self) -> None:
        assert is_valid_url("http://shop.example.com") is True

    def test_missing_scheme(self) -> None:
        assert is_valid_url("example.com") is False

    def test_ftp_scheme(self) -> None:
        assert is_valid_url("ftp://example.com") is False

    def test_empty_string(self) -> None:
        assert is_valid_url("") is False


class TestNormaliseUrl:
    """Tests for ``normalise_url``."""

    def test_adds_scheme(self) -> None:
        assert normalise_url("example.com") == "https://example.com"

    def test_strips_whitespace(self) -> None:
        assert normalise_url("  https://example.com  ") == "https://example.com"

    def test_preserves_http(self) -> None:
        assert normalise_url("http://example.com") == "http://example.com"


class TestSanitiseText:
    """Tests for ``sanitise_text``."""

    def test_strips_whitespace(self) -> None:
        assert sanitise_text("  hello world  ") == "hello world"

    def test_collapses_spaces(self) -> None:
        assert sanitise_text("a   b\n\tc") == "a b c"

    def test_unescapes_html(self) -> None:
        assert sanitise_text("Tom &amp; Jerry") == "Tom & Jerry"

    def test_none_returns_na(self) -> None:
        assert sanitise_text(None) == "N/A"

    def test_empty_returns_na(self) -> None:
        assert sanitise_text("") == "N/A"


class TestSanitisePrice:
    """Tests for ``sanitise_price``."""

    def test_dollar(self) -> None:
        assert sanitise_price("$19.99") == "$19.99"

    def test_euro(self) -> None:
        assert sanitise_price("€29,50") == "€29,50"

    def test_none_returns_na(self) -> None:
        assert sanitise_price(None) == "N/A"


class TestSafeFilename:
    """Tests for ``safe_filename``."""

    def test_removes_invalid_chars(self) -> None:
        assert safe_filename('a<b>c:d"e') == "a_b_c_d_e"

    def test_truncates(self) -> None:
        result = safe_filename("x" * 300, max_length=10)
        assert len(result) == 10

    def test_empty(self) -> None:
        assert safe_filename("") == "unnamed"


class TestDetectCaptcha:
    """Tests for ``detect_captcha``."""

    def test_captcha_detected(self) -> None:
        assert detect_captcha("<html><body>Please complete the CAPTCHA</body></html>") is True

    def test_clean_page(self) -> None:
        assert detect_captcha("<html><body>Products</body></html>") is False

    def test_access_denied(self) -> None:
        assert detect_captcha("<html><title>Access Denied</title></html>") is True


# ═══════════════════════════════════════════════════════════════
# AntiBot tests
# ═══════════════════════════════════════════════════════════════

class TestAntiBot:
    """Tests for ``AntiBot``."""

    def test_requires_at_least_one_ua(self) -> None:
        with pytest.raises(ValueError):
            AntiBot(user_agents=[])

    def test_random_user_agent(self) -> None:
        uas = ["UA1", "UA2", "UA3"]
        ab = AntiBot(uas, delay_min=0, delay_max=0)
        assert ab.random_user_agent() in uas

    def test_headers_contain_ua(self) -> None:
        ab = AntiBot(["TestAgent"], delay_min=0, delay_max=0)
        headers = ab.get_headers()
        assert headers["User-Agent"] == "TestAgent"

    def test_extra_headers_merged(self) -> None:
        ab = AntiBot(["TestAgent"], delay_min=0, delay_max=0)
        headers = ab.get_headers(extra={"X-Custom": "value"})
        assert headers["X-Custom"] == "value"


# ═══════════════════════════════════════════════════════════════
# Paginator tests
# ═══════════════════════════════════════════════════════════════

class TestPaginator:
    """Tests for ``Paginator``."""

    def test_build_query_param_url(self) -> None:
        p = Paginator(page_param="page")
        url = p.build_page_url("https://shop.com/products", 3)
        assert "page=3" in url

    def test_build_path_url(self) -> None:
        p = Paginator(pagination_type="path")
        url = p.build_page_url("https://shop.com/products", 5, "path")
        assert "/page/5/" in url

    def test_find_next_url_present(self) -> None:
        html_str = '<html><body><a class="next-page" href="/page/2">Next</a></body></html>'
        soup = BeautifulSoup(html_str, "html.parser")
        p = Paginator(next_page_selector="a.next-page")
        result = p.find_next_url(soup, "https://shop.com")
        assert result == "https://shop.com/page/2"

    def test_find_next_url_absent(self) -> None:
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        p = Paginator()
        assert p.find_next_url(soup) is None


# ═══════════════════════════════════════════════════════════════
# ProductScraper parse tests
# ═══════════════════════════════════════════════════════════════

class TestProductScraperParsing:
    """Tests for ``ProductScraper.parse_products`` using fixture HTML."""

    _FIXTURE = textwrap.dedent("""\
        <html><body>
        <div class="product-item">
            <h2 class="product-title">Cool T-Shirt</h2>
            <span class="price">$29.99</span>
            <p class="product-description">A very cool shirt.</p>
            <img class="product-image" src="https://img.example.com/tshirt.jpg"/>
            <a href="https://shop.example.com/cool-tshirt">View</a>
        </div>
        <div class="product-item">
            <h2 class="product-title">Fancy Hat</h2>
            <span class="price">€15,50</span>
            <a href="/fancy-hat">View</a>
        </div>
        </body></html>
    """)

    @pytest.fixture()
    def scraper(self, tmp_path: Path) -> ProductScraper:
        """Create a scraper with default selectors (no config file)."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("selectors:\n  product_container: div.product-item\n")
        return ProductScraper(config_path=str(cfg))

    def test_extracts_two_products(self, scraper: ProductScraper) -> None:
        soup = BeautifulSoup(self._FIXTURE, "lxml")
        products = scraper.parse_products(soup, "https://shop.example.com")
        assert len(products) == 2

    def test_first_product_fields(self, scraper: ProductScraper) -> None:
        soup = BeautifulSoup(self._FIXTURE, "lxml")
        products = scraper.parse_products(soup, "https://shop.example.com")
        p = products[0]
        assert p["name"] == "Cool T-Shirt"
        assert "$29.99" in p["price"] or "29.99" in p["price"]
        assert p["description"] == "A very cool shirt."
        assert p["image_url"] == "https://img.example.com/tshirt.jpg"

    def test_missing_fields_default_to_na(self, scraper: ProductScraper) -> None:
        soup = BeautifulSoup(self._FIXTURE, "lxml")
        products = scraper.parse_products(soup, "https://shop.example.com")
        p = products[1]
        # Second product has no image and no description
        assert p["description"] == "N/A"
        assert p["image_url"] == "N/A"

"""Input validation and data-sanitisation helpers.

Functions in this module are **pure** (no side-effects) and safe to call from
any thread.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlparse

from utils.logger import get_logger

log = get_logger(__name__)


# ── URL validation ──────────────────────────────────────────────

def is_valid_url(url: str) -> bool:
    """Return *True* if *url* looks like a valid HTTP(S) URL.

    Args:
        url: The URL string to validate.

    Returns:
        ``True`` when the URL has an ``http`` or ``https`` scheme **and** a
        non-empty network location (domain).
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:  # noqa: BLE001
        return False


def normalise_url(url: str) -> str:
    """Strip whitespace and ensure the URL has a scheme.

    Args:
        url: Raw user-provided URL.

    Returns:
        A trimmed URL with ``https://`` prepended when no scheme is present.
    """
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


# ── Text sanitisation ──────────────────────────────────────────

_MULTI_SPACE = re.compile(r"\s+")


def sanitise_text(text: str | None) -> str:
    """Clean scraped text for storage.

    * Strips leading/trailing whitespace.
    * Collapses consecutive whitespace to a single space.
    * Un-escapes HTML entities (``&amp;`` → ``&``).
    * Returns ``"N/A"`` for empty / ``None`` input.

    Args:
        text: Raw text from an HTML element.

    Returns:
        Cleaned string, or ``"N/A"``.
    """
    if not text:
        return "N/A"
    text = html.unescape(text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text if text else "N/A"


def sanitise_price(raw: str | None) -> str:
    """Extract a human-readable price string.

    Keeps digits, dots, commas, currency symbols, and whitespace.

    Args:
        raw: Raw price text (e.g. ``"  $19.99  "``).

    Returns:
        Cleaned price or ``"N/A"``.
    """
    if not raw:
        return "N/A"
    raw = html.unescape(raw).strip()
    # Keep currency symbols, digits, dots, commas, spaces
    cleaned = re.sub(r"[^\d.,€£$¥₹\s]", "", raw).strip()
    return cleaned if cleaned else "N/A"


def safe_filename(name: str, max_length: int = 200) -> str:
    """Convert an arbitrary string into a filesystem-safe filename.

    Args:
        name: Original string.
        max_length: Maximum character length for the result.

    Returns:
        Sanitised filename fragment (no extension).
    """
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.strip(". ")
    if len(name) > max_length:
        name = name[:max_length]
    return name if name else "unnamed"


def detect_captcha(html_content: str) -> bool:
    """Heuristically detect CAPTCHA / access-denied pages.

    Args:
        html_content: The raw HTML body of a response.

    Returns:
        ``True`` when the page likely contains a CAPTCHA challenge.
    """
    indicators = [
        "captcha",
        "robot check",
        "access denied",
        "please verify you are a human",
        "unusual traffic",
        "are you a robot",
        "challenge-platform",
        "cf-challenge",
        "recaptcha",
        "hcaptcha",
    ]
    lower = html_content.lower()
    for indicator in indicators:
        if indicator in lower:
            log.warning("CAPTCHA indicator detected: '%s'", indicator)
            return True
    return False

"""TLS-impersonation fetcher using ``curl_cffi``.

Many modern WAFs (Akamai Bot Manager, Cloudflare, DataDome) fingerprint
the **TLS ClientHello** sent by the HTTP client.  Standard ``requests``
presents a Python/OpenSSL fingerprint that is trivial to detect.

``curl_cffi`` bundles a patched libcurl that can impersonate the TLS
fingerprint of real browsers (Chrome, Firefox, Safari, Edge).  This
often bypasses WAFs *without* needing a full headless browser.

Requires::

    pip install curl_cffi

Usage::

    from core.tls_fetcher import TLSFetcher

    fetcher = TLSFetcher()
    html = fetcher.fetch("https://protected-site.com/products")
"""

from __future__ import annotations

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from curl_cffi import requests as cffi_requests

    _CFFI_AVAILABLE = True
except ImportError:
    _CFFI_AVAILABLE = False
    log.debug("curl_cffi not installed — TLS-impersonation unavailable")


# Impersonation targets to try, in order of preference.
# Modern Chrome is the safest bet; we fall back to Firefox & Safari.
_IMPERSONATION_TARGETS: list[str] = [
    "chrome",      # latest Chrome
    "chrome120",   # specific version
    "firefox",     # Firefox
    "safari",      # Safari
    "edge",        # Edge
]

# Realistic headers that complement the TLS fingerprint.
_DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "max-age=0",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Linux"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


class TLSFetcher:
    """Fetch pages with a realistic TLS fingerprint via ``curl_cffi``.

    Args:
        timeout: Request timeout in seconds.
    """

    def __init__(self, timeout: int = 30) -> None:
        if not _CFFI_AVAILABLE:
            raise RuntimeError(
                "curl_cffi is not installed.  Run:\n  pip install curl_cffi"
            )
        self._timeout = timeout
        self._session: cffi_requests.Session | None = None

    def _ensure_session(self) -> cffi_requests.Session:
        if self._session is None:
            self._session = cffi_requests.Session()
        return self._session

    def fetch(self, url: str) -> str | None:
        """Fetch *url* trying multiple browser impersonation targets.

        Returns the HTML body on success, or ``None`` if all targets fail
        or the response looks like a WAF block page.
        """
        session = self._ensure_session()

        for target in _IMPERSONATION_TARGETS:
            try:
                log.debug("TLS fetch %s (impersonate=%s)", url, target)
                resp = session.get(
                    url,
                    headers=_DEFAULT_HEADERS,
                    impersonate=target,
                    timeout=self._timeout,
                    allow_redirects=True,
                )

                if resp.status_code == 403:
                    log.debug("TLS/%s → 403, trying next target", target)
                    continue

                if resp.status_code >= 400:
                    log.debug("TLS/%s → HTTP %d", target, resp.status_code)
                    continue

                html = resp.text

                # Quick check: is this a WAF block page?
                if self._looks_like_block(html):
                    log.debug("TLS/%s → block page detected, trying next", target)
                    continue

                log.info(
                    "TLS fetch OK (%s): HTTP %d, %d bytes",
                    target, resp.status_code, len(html),
                )
                return html

            except Exception as exc:  # noqa: BLE001
                log.debug("TLS/%s failed: %s", target, exc)
                continue

        log.warning("TLS fetch failed for %s (all impersonation targets exhausted)", url)
        return None

    @staticmethod
    def _looks_like_block(html: str) -> bool:
        """Return ``True`` if *html* looks like a WAF interstitial."""
        lower = html.lower()
        # Akamai behavioral challenge page (JS challenge with sensor data)
        if "sec-if-cpt-container" in html or "behavioral-content" in lower:
            return True
        if "scf-akamai-logo" in html:
            return True
        if len(html) < 2500:
            for indicator in ("access denied", "robot check", "captcha",
                              "just a moment", "checking your browser"):
                if indicator in lower:
                    return True
        if "_abck" in html and "akam" in lower and len(html) < 5000:
            return True
        return False

    def close(self) -> None:
        """Close the underlying session."""
        if self._session is not None:
            self._session.close()
            self._session = None
            log.debug("TLS session closed")


def is_curl_cffi_available() -> bool:
    """Return ``True`` if ``curl_cffi`` is importable."""
    return _CFFI_AVAILABLE

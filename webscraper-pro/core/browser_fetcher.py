"""Playwright-based page fetcher for JS-rendered / WAF-protected sites.

Falls back to a real headless browser when ``requests`` gets blocked
(403/CAPTCHA) by services like Akamai, Cloudflare, DataDome, etc.

Strategy:
1. **Headed Firefox + Xvfb** (best) — the browser thinks it has a real
   display, which makes canvas/WebGL fingerprinting return genuine values.
2. **Headless Firefox** — still decent TLS fingerprint; some WAFs let it
   through.
3. **Chromium** (system or bundled) — last resort.

Requires::

    pip install playwright
    python -m playwright install firefox   # preferred
    # Optional for headed mode without a display:
    sudo apt-get install xvfb

Usage::

    from core.browser_fetcher import BrowserFetcher

    fetcher = BrowserFetcher()
    html = fetcher.fetch("https://protected-site.com/products")
    fetcher.close()
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from playwright.sync_api import Browser, Page, Playwright, sync_playwright

    _PW_AVAILABLE = True
except ImportError:
    _PW_AVAILABLE = False
    log.debug("Playwright not installed — browser fetcher unavailable")


# ── Stealth init-script injected into every browser context ─────
# Patches many browser properties that WAFs fingerprint to detect
# headless / automated browsers.
_STEALTH_JS = """
(() => {
    // 1. navigator.webdriver → undefined
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

    // 2. navigator.plugins — headless Chromium has 0 plugins
    if (navigator.plugins.length === 0) {
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
    }

    // 3. navigator.languages
    Object.defineProperty(navigator, 'languages', {
        get: () => ['it-IT', 'it', 'en-US', 'en'],
    });

    // 4. Permissions API — hide "denied" for notifications
    if (navigator.permissions) {
        const origQuery = navigator.permissions.query;
        navigator.permissions.query = (params) =>
            params.name === 'notifications'
                ? Promise.resolve({state: Notification.permission})
                : origQuery.call(navigator.permissions, params);
    }

    // 5. chrome.runtime — headless Chromium is missing this
    if (!window.chrome) window.chrome = {};
    if (!window.chrome.runtime) window.chrome.runtime = {};
})();
"""


class BrowserFetcher:
    """Headless browser fetcher using Playwright.

    The browser is lazily started on the first call to :meth:`fetch` and
    reused for subsequent requests.  Call :meth:`close` when done.

    The launch strategy is:

    1. If ``DISPLAY`` is not set and ``Xvfb`` is available, start a
       virtual display and launch Firefox in **headed mode** (most
       realistic fingerprint).
    2. Otherwise, launch Firefox in headless mode.
    3. If Firefox is unavailable, fall back to Chromium.

    Args:
        headless: Run the browser without a visible window.  If ``None``
            (default), the class auto-selects headed mode when Xvfb is
            available and there is no physical display.
        timeout: Navigation timeout in milliseconds.
    """

    def __init__(self, headless: bool | None = None, timeout: int = 45_000) -> None:
        if not _PW_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed.  Run:\n"
                "  pip install playwright && python -m playwright install chromium"
            )
        self._headless_pref = headless          # None = auto
        self._timeout = timeout
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._xvfb_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # ── Xvfb management ────────────────────────────────────────

    @staticmethod
    def _has_display() -> bool:
        return bool(os.environ.get("DISPLAY"))

    @staticmethod
    def _xvfb_available() -> bool:
        return shutil.which("Xvfb") is not None

    def _start_xvfb(self) -> bool:
        """Start Xvfb on display :99 if not already running."""
        if self._has_display():
            return True  # a display is already available
        if not self._xvfb_available():
            return False
        try:
            self._xvfb_proc = subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.environ["DISPLAY"] = ":99"
            time.sleep(0.5)  # let Xvfb initialize
            log.info("Started Xvfb virtual display on :99")
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not start Xvfb: %s", exc)
            return False

    def _stop_xvfb(self) -> None:
        if self._xvfb_proc is not None:
            self._xvfb_proc.terminate()
            self._xvfb_proc.wait(timeout=5)
            self._xvfb_proc = None
            log.debug("Xvfb stopped")

    # ── Lazy browser init ──────────────────────────────────────

    def _decide_headless(self) -> bool:
        """Choose headed vs headless mode.

        If the user explicitly set headless, honour that.  Otherwise:
        - If a display exists (DISPLAY or Xvfb) → headed mode (best stealth)
        - Otherwise → headless mode
        """
        if self._headless_pref is not None:
            return self._headless_pref

        # Try to get a display (real or virtual)
        if self._has_display() or self._start_xvfb():
            log.info("Auto-selected HEADED mode (Xvfb virtual display)")
            return False  # headed
        return True  # headless

    def _ensure_browser(self) -> Browser:
        if self._browser is not None:
            return self._browser
        with self._lock:
            if self._browser is not None:
                return self._browser
            headless = self._decide_headless()
            self._pw = sync_playwright().start()

            for launcher in (
                self._try_firefox,
                self._try_chromium_channel,
                self._try_chromium_bundled,
            ):
                browser = launcher(headless)
                if browser:
                    self._browser = browser
                    return self._browser

            raise RuntimeError(
                "Could not launch any browser. Install Firefox or Chromium:\n"
                "  python -m playwright install firefox\n"
                "  sudo apt-get install libgtk-3-0t64 libdbus-glib-1-2"
            )

    def _try_firefox(self, headless: bool) -> Browser | None:
        try:
            assert self._pw is not None
            browser = self._pw.firefox.launch(headless=headless)
            log.info("Launched Firefox (headless=%s)", headless)
            return browser
        except Exception as exc:  # noqa: BLE001
            log.debug("Firefox launch failed: %s", exc)
            return None

    def _try_chromium_channel(self, headless: bool) -> Browser | None:
        try:
            assert self._pw is not None
            browser = self._pw.chromium.launch(
                headless=headless,
                channel="chromium",
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            log.info("Launched system Chromium (headless=%s)", headless)
            return browser
        except Exception as exc:  # noqa: BLE001
            log.debug("System-chromium launch failed: %s", exc)
            return None

    def _try_chromium_bundled(self, headless: bool) -> Browser | None:
        try:
            assert self._pw is not None
            browser = self._pw.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            log.info("Launched bundled Chromium (headless=%s)", headless)
            return browser
        except Exception as exc:  # noqa: BLE001
            log.debug("Bundled-chromium launch failed: %s", exc)
            return None

    # ── Public API ──────────────────────────────────────────────

    def fetch(self, url: str) -> str | None:
        """Navigate to *url* and return the rendered HTML.

        Creates a context with a realistic viewport / user-agent and waits
        for ``domcontentloaded``.  If a WAF challenge page is detected,
        waits up to 20 s for the challenge to resolve, then tries a reload.

        Args:
            url: Absolute URL to load.

        Returns:
            Full page HTML, or ``None`` on failure.
        """
        try:
            browser = self._ensure_browser()

            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                locale="it-IT",
                timezone_id="Europe/Rome",
                extra_http_headers={
                    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
                },
            )
            context.add_init_script(_STEALTH_JS)

            page: Page = context.new_page()
            log.debug("Browser navigating → %s", url)

            response = page.goto(
                url, wait_until="domcontentloaded", timeout=self._timeout,
            )
            page.wait_for_timeout(3000)

            html = page.content()
            status = response.status if response else 0

            # ── Handle WAF challenge pages ──────────────────────
            if self._looks_like_challenge(html):
                log.info("Detected WAF challenge page — waiting for auto-solve …")
                html = self._wait_for_challenge(page)

            context.close()

            if self._looks_like_challenge(html):
                log.warning(
                    "Browser still blocked for %s after challenge wait "
                    "(site uses aggressive bot protection)",
                    url,
                )
                return None

            log.info(
                "Browser fetched %s (HTTP %d, %d bytes)",
                url, status, len(html),
            )
            return html

        except Exception as exc:  # noqa: BLE001
            log.error("Browser fetch failed for %s: %s", url, exc)
            return None

    def _wait_for_challenge(self, page: Page) -> str:
        """Wait for a WAF challenge to resolve in-page.

        The typical Akamai / Cloudflare flow is:
        1. Challenge script collects sensor data (mouse, keyboard, device)
        2. Sends it to the WAF server
        3. Gets the OK cookie
        4. Reloads the page → real content served

        We simulate realistic user behaviour (mouse moves, scroll)
        so the sensor data collection script thinks we're a real user.
        """
        import random

        # Simulate human-like interactions that generate sensor data
        def _simulate_user() -> None:
            try:
                # Random mouse movements across the page
                for _ in range(random.randint(3, 6)):
                    x = random.randint(100, 1800)
                    y = random.randint(100, 900)
                    page.mouse.move(x, y)
                    page.wait_for_timeout(random.randint(150, 400))

                # Small scroll
                page.mouse.wheel(0, random.randint(50, 200))
                page.wait_for_timeout(random.randint(300, 700))

                # Move to center-ish area
                page.mouse.move(
                    random.randint(400, 1200),
                    random.randint(300, 600),
                )
            except Exception:  # noqa: BLE001
                pass

        for attempt in range(4):
            _simulate_user()

            try:
                page.wait_for_load_state("load", timeout=10_000)
                page.wait_for_timeout(2000)
                html = page.content()
                if not self._looks_like_challenge(html):
                    log.info("Challenge solved on attempt %d", attempt + 1)
                    return html
            except Exception:  # noqa: BLE001
                pass

            # Manually reload to check if cookies are now set
            try:
                log.debug("Manual reload (attempt %d)", attempt + 1)
                page.reload(wait_until="domcontentloaded", timeout=self._timeout)
                page.wait_for_timeout(3000)
                html = page.content()
                if not self._looks_like_challenge(html):
                    log.info("Challenge solved after reload (attempt %d)", attempt + 1)
                    return html
            except Exception:  # noqa: BLE001
                pass

        return page.content()

    @staticmethod
    def _looks_like_challenge(html: str) -> bool:
        """Detect WAF challenge / access-denied pages."""
        lower = html.lower()
        if "sec-if-cpt-container" in html or "behavioral-content" in lower:
            return True
        if "scf-akamai-logo" in html:
            return True
        if len(html) < 2000:
            for ind in ("access denied", "robot check", "captcha"):
                if ind in lower:
                    return True
        if "_abck" in html and "akam" in lower and len(html) < 5000:
            return True
        # Cloudflare challenge
        if "cf-browser-verification" in lower and len(html) < 10_000:
            return True
        return False

    def close(self) -> None:
        """Shut down the browser, Playwright, and Xvfb."""
        with self._lock:
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._pw:
                self._pw.stop()
                self._pw = None
                log.debug("Playwright stopped")
            self._stop_xvfb()


def is_playwright_available() -> bool:
    """Return ``True`` if Playwright is importable."""
    return _PW_AVAILABLE

"""Anti-bot countermeasures: User-Agent rotation and request delay.

This module is intentionally kept small so it can be unit-tested in
isolation.
"""

from __future__ import annotations

import random
import time
from typing import Sequence

from utils.logger import get_logger

log = get_logger(__name__)


class AntiBot:
    """Provides request-level anti-bot helpers.

    Args:
        user_agents: List of User-Agent strings to rotate through.
        delay_min: Minimum delay between requests (seconds).
        delay_max: Maximum delay between requests (seconds).
    """

    def __init__(
        self,
        user_agents: Sequence[str],
        delay_min: float = 1.5,
        delay_max: float = 3.5,
    ) -> None:
        if not user_agents:
            raise ValueError("At least one User-Agent string is required")
        self._user_agents: list[str] = list(user_agents)
        self._delay_min = delay_min
        self._delay_max = delay_max

    # ── Public API ──────────────────────────────────────────────

    def random_user_agent(self) -> str:
        """Return a randomly chosen User-Agent string.

        Returns:
            A User-Agent header value.
        """
        ua = random.choice(self._user_agents)  # noqa: S311
        log.debug("Selected User-Agent: %s", ua[:60])
        return ua

    def random_delay(self) -> float:
        """Sleep for a random duration within the configured range.

        Returns:
            The actual number of seconds slept.
        """
        delay = random.uniform(self._delay_min, self._delay_max)  # noqa: S311
        log.debug("Sleeping %.2f s …", delay)
        time.sleep(delay)
        return delay

    def get_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build a plausible request headers dict.

        Args:
            extra: Additional headers to merge.

        Returns:
            Headers dict with a rotated User-Agent.
        """
        headers: dict[str, str] = {
            "User-Agent": self.random_user_agent(),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",
        }
        if extra:
            headers.update(extra)
        return headers

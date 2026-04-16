"""
BaseAgent — shared foundation for every agent in the pipeline.

Responsibilities handled here so that concrete agents stay lean:
  • OpenAI-compatible client pointed at OpenRouter
  • Automatic rate-limit pauses between calls
  • Retry loop with exponential back-off on transient errors (429, 5xx)
  • Structured logging
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI, APIError, AuthenticationError, NotFoundError, RateLimitError, APIConnectionError

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_APP_NAME,
    OPENROUTER_SITE_URL,
    MAX_TOKENS,
    RATE_LIMIT_PAUSE,
    MAX_RETRIES,
)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Subclasses MUST define:
      • name            – human-readable agent label
      • model           – OpenRouter model identifier (primary)
      • fallback_models – list of fallback model identifiers
      • system_prompt   – instructions that shape the agent's behaviour
      • temperature     – sampling temperature (creative vs. analytical)

    And implement:
      • run(**kwargs) -> str
    """

    name: str = "BaseAgent"
    model: str = ""
    fallback_models: list[str] = []
    system_prompt: str = ""
    temperature: float = 0.7

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.name)
        self._active_model: str = self.model  # tracks which model is in use

        # The openai client works with any OpenAI-compatible API
        # when you override base_url and supply the right key.
        self._client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": OPENROUTER_SITE_URL or "http://localhost",
                "X-Title": OPENROUTER_APP_NAME,
            },
        )

    # ── Core LLM call with rotate-first, retry-after logic ──

    def _call_llm(self, user_message: str, *, max_tokens: int = MAX_TOKENS) -> str:
        """
        Send a chat-completion request to OpenRouter.

        Strategy — **rotate first, retry after**:
          1. On 429 (rate-limit) or transient error → immediately try the
             NEXT model in the chain instead of waiting on the same one.
          2. On 404 (model removed) → permanently skip that model.
          3. Only after every model in the chain has been tried once do we
             start a new round (up to MAX_RETRIES rounds total).
          4. 401 (bad key) → fail immediately, no fallback.

        This minimises wasted wait time on the free tier: instead of
        burning 5→10→20→40→80 s on one rate-limited model, we rotate
        through 4-5 different models in seconds.

        Returns the assistant's text response.
        """
        # Build the full model chain: primary + fallbacks (deduplicated)
        model_chain = [self.model] + [
            fb for fb in self.fallback_models if fb != self.model
        ]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Models that returned 404 — permanently dead, never retry
        dead_models: set[str] = set()
        last_error: Exception | None = None

        for round_num in range(1, MAX_RETRIES + 1):
            alive_models = [m for m in model_chain if m not in dead_models]
            if not alive_models:
                break

            for current_model in alive_models:
                # Respect free-tier rate limits
                self.logger.debug("Rate-limit pause (%.1fs)...", RATE_LIMIT_PAUSE)
                time.sleep(RATE_LIMIT_PAUSE)

                try:
                    self.logger.info(
                        "→ [%s] model=%s  temp=%.2f  round=%d/%d",
                        self.name, current_model, self.temperature,
                        round_num, MAX_RETRIES,
                    )

                    response = self._client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=max_tokens,
                    )

                    text = response.choices[0].message.content or ""
                    usage = response.usage
                    if usage:
                        self.logger.info(
                            "← [%s] tokens_in=%d  tokens_out=%d",
                            self.name,
                            usage.prompt_tokens,
                            usage.completion_tokens,
                        )

                    # Remember which model actually worked
                    self._active_model = current_model
                    return text.strip()

                except AuthenticationError as exc:
                    # 401 — bad API key, no point retrying or falling back
                    raise RuntimeError(
                        f"[{self.name}] Authentication failed (401). "
                        "Check your OPENROUTER_API_KEY in .env. "
                        "Get a free key at https://openrouter.ai/keys"
                    ) from exc

                except NotFoundError:
                    # 404 — model gone, mark dead and move on
                    self.logger.warning(
                        "Model not found (404): %s — removed from chain.",
                        current_model,
                    )
                    dead_models.add(current_model)
                    last_error = NotFoundError(
                        f"Model not found: {current_model}"
                    )
                    continue  # next model in this round

                except RateLimitError as exc:
                    # 429 — DON'T wait, rotate to next model immediately
                    self.logger.warning(
                        "Rate-limited (429) on %s — rotating to next model.",
                        current_model,
                    )
                    last_error = exc
                    continue  # next model in this round

                except (APIError, APIConnectionError) as exc:
                    # 5xx / network blip — rotate immediately
                    self.logger.warning(
                        "API error on %s (%s) — rotating to next model.",
                        current_model, exc,
                    )
                    last_error = exc
                    continue  # next model in this round

            # ── End of one full rotation through all alive models ────
            # If we get here, every alive model failed this round.
            # Wait with exponential back-off before starting a new round.
            if round_num < MAX_RETRIES:
                wait = 5 * (2 ** (round_num - 1))  # 5s, 10s, 20s, 40s
                self.logger.warning(
                    "All models failed round %d/%d. "
                    "Waiting %ds before next round...",
                    round_num, MAX_RETRIES, wait,
                )
                time.sleep(wait)

        # All rounds exhausted
        tried = ", ".join(m for m in model_chain if m not in dead_models)
        skipped = ", ".join(dead_models) if dead_models else "none"
        raise RuntimeError(
            f"[{self.name}] All {MAX_RETRIES} rounds failed. "
            f"Tried: {tried}. Skipped (404): {skipped}. "
            f"Last error: {last_error}"
        )

    # ── Public interface (implemented by subclasses) ────────

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the agent's task and return the textual result."""
        ...

    def __repr__(self) -> str:
        return f"<{self.name} model={self.model}>"

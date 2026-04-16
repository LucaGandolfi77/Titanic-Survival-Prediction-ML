"""
text_processor.py — LLM-powered text cleaner for TTS narration.

Uses free OpenRouter models to:
  • Remove OCR artifacts, page numbers, headers/footers
  • Expand abbreviations
  • Normalise punctuation for natural speech pauses
  • Optionally simplify long sentences

Uses the same rotate-first fallback strategy from the openrouter project.
"""

from __future__ import annotations

import logging
import time

from openai import (
    OpenAI,
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    APIConnectionError,
)

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_APP_NAME,
    MODEL_CLEANER,
    FALLBACK_MODELS,
    TEMPERATURE,
    MAX_TOKENS,
    RATE_LIMIT_PAUSE,
    MAX_RETRIES,
    CLEAN_MAX_CHUNK_CHARS,
)

logger = logging.getLogger("TextProcessor")

SYSTEM_PROMPT = (
    "You are a text-preparation assistant for a Text-to-Speech system.\n\n"
    "You will receive a raw excerpt from a book. Clean it up for narration:\n"
    "1. Remove page numbers, headers, footers, and repeated watermarks.\n"
    "2. Remove any OCR artifacts (broken words, stray characters).\n"
    "3. Expand abbreviations (Dr. → Doctor, Mr. → Mister, etc.).\n"
    "4. Write out numbers under 100 as words (42 → forty-two).\n"
    "5. Normalise quotes and dashes to simple ASCII forms.\n"
    "6. Keep ALL original content — do NOT summarise or skip paragraphs.\n"
    "7. Preserve paragraph breaks as double newlines.\n"
    "8. Do NOT add any commentary — return ONLY the cleaned text.\n"
)


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": OPENROUTER_APP_NAME,
        },
    )


def _call_llm(client: OpenAI, text_chunk: str) -> str:
    """
    Call OpenRouter with rotate-first fallback on 429.
    """
    model_chain = [MODEL_CLEANER] + [
        m for m in FALLBACK_MODELS if m != MODEL_CLEANER
    ]
    dead: set[str] = set()
    last_error: Exception | None = None

    for round_num in range(1, MAX_RETRIES + 1):
        alive = [m for m in model_chain if m not in dead]
        if not alive:
            break

        for model in alive:
            logger.debug("Rate-limit pause (%.1fs)...", RATE_LIMIT_PAUSE)
            time.sleep(RATE_LIMIT_PAUSE)

            try:
                logger.info("→ [Cleaner] model=%s  round=%d/%d", model, round_num, MAX_RETRIES)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text_chunk},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                if usage:
                    logger.info("← [Cleaner] tokens_in=%d  tokens_out=%d",
                                usage.prompt_tokens, usage.completion_tokens)
                return text.strip()

            except AuthenticationError as exc:
                raise RuntimeError(
                    "Authentication failed (401). Check OPENROUTER_API_KEY."
                ) from exc
            except NotFoundError:
                logger.warning("Model not found (404): %s — removed.", model)
                dead.add(model)
                last_error = NotFoundError(f"404: {model}")
                continue
            except RateLimitError as exc:
                logger.warning("429 on %s — rotating.", model)
                last_error = exc
                continue
            except (APIError, APIConnectionError) as exc:
                logger.warning("API error on %s — rotating.", model)
                last_error = exc
                continue

        # All alive models failed this round
        if round_num < MAX_RETRIES:
            wait = 5 * (2 ** (round_num - 1))
            logger.warning("All models failed round %d. Waiting %ds...", round_num, wait)
            time.sleep(wait)

    raise RuntimeError(f"Text cleaning failed after {MAX_RETRIES} rounds. Last: {last_error}")


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def clean_text(text: str) -> str:
    """
    Clean the full book text for TTS narration using an LLM.

    If no API key is set, returns the text as-is.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("No OPENROUTER_API_KEY — skipping LLM cleaning.")
        return text

    client = _make_client()
    chunks = _chunk_text(text, CLEAN_MAX_CHUNK_CHARS)
    logger.info("Cleaning text: %d chunks of ≤%d chars each.", len(chunks), CLEAN_MAX_CHUNK_CHARS)

    cleaned: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info("Cleaning chunk %d / %d (%d chars)...", i, len(chunks), len(chunk))
        result = _call_llm(client, chunk)
        cleaned.append(result)

    return "\n\n".join(cleaned)

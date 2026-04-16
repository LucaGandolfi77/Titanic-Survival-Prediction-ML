"""
Agent 4 — The Translator.

Receives the full book text (all 20 chapters concatenated) and
translates it into Italian, preserving formatting and literary style.
"""

from __future__ import annotations

from typing import Any

from config import MODEL_TRANSLATOR, FALLBACK_TRANSLATOR, TEMPERATURE_ANALYTICAL
from .base import BaseAgent


class TranslatorAgent(BaseAgent):
    name = "Translator"
    model = MODEL_TRANSLATOR
    fallback_models = FALLBACK_TRANSLATOR
    temperature = TEMPERATURE_ANALYTICAL

    system_prompt = (
        "You are an expert literary translator specializing in English-to-Italian "
        "translation.\n\n"
        "You receive a chapter of a book in English and must translate it into "
        "fluent, natural Italian.\n\n"
        "RULES:\n"
        "1. Preserve the original markdown formatting (headings, paragraphs, etc.).\n"
        "2. Maintain the literary style, tone, and voice of the original.\n"
        "3. Translate character names only if they have an obvious Italian equivalent; "
        "otherwise keep them as-is.\n"
        "4. Translate dialogue naturally — it should sound like native Italian speech.\n"
        "5. Do NOT add, remove, or summarize any content. The translation must be "
        "complete and faithful.\n"
        "6. Return ONLY the translated text — no preamble, no notes."
    )

    def run(self, *, chapter_text: str, chapter_number: int = 0, **kwargs: Any) -> str:
        """
        Translate a single chapter from English to Italian.

        Args:
            chapter_text: the English chapter text.
            chapter_number: chapter number (for logging).

        Returns:
            The full Italian translation of the chapter.
        """
        prompt = (
            "Translate the following chapter into Italian:\n\n"
            f"{chapter_text}\n\n"
            "Provide the complete Italian translation now."
        )

        self.logger.info("Translating chapter %d to Italian...", chapter_number)
        translation = self._call_llm(prompt)
        self.logger.info(
            "Translation complete (%d chars → %d chars).",
            len(chapter_text), len(translation),
        )
        return translation

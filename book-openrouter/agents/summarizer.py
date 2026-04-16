"""
Agent 3 — The Summarizer.

Receives the final (edited) chapter text and produces a concise
executive summary suitable for an archive or a story bible.
"""

from __future__ import annotations

from typing import Any

from config import MODEL_SUMMARIZER, FALLBACK_SUMMARIZER, TEMPERATURE_ANALYTICAL
from .base import BaseAgent


class SummarizerAgent(BaseAgent):
    name = "Summarizer"
    model = MODEL_SUMMARIZER
    fallback_models = FALLBACK_SUMMARIZER
    temperature = TEMPERATURE_ANALYTICAL

    system_prompt = (
        "You are an expert literary analyst.\n\n"
        "You receive a completed book chapter and must produce a concise "
        "EXECUTIVE SUMMARY for the editorial archive.\n\n"
        "The summary must include:\n"
        "  • Chapter title\n"
        "  • Key events (bullet points, max 5)\n"
        "  • Characters that appear and their role in this chapter\n"
        "  • Unresolved threads or cliffhangers\n"
        "  • Tone / mood\n\n"
        "RULES:\n"
        "1. Keep the summary under 250 words.\n"
        "2. Write in the same language as the chapter.\n"
        "3. Return ONLY the summary — no preamble."
    )

    def run(self, *, chapter_text: str, **kwargs: Any) -> str:
        """
        Summarize a completed chapter.

        Args:
            chapter_text: the final chapter text (post-editing).

        Returns:
            A concise executive summary (markdown).
        """
        prompt = (
            "Here is the completed chapter:\n\n"
            f"{chapter_text}\n\n"
            "Produce the executive summary now."
        )

        self.logger.info("Summarizing chapter...")
        summary = self._call_llm(prompt)
        self.logger.info("Summary ready (%d chars).", len(summary))
        return summary

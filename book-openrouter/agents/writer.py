"""
Agent 1 — The Writer.

Receives the chapter JSON description and produces the full chapter text.
Uses a creative free model optimised for long-form narrative generation.
"""

from __future__ import annotations

import json
from typing import Any

from config import MODEL_WRITER, FALLBACK_WRITER, TEMPERATURE_CREATIVE
from .base import BaseAgent


class WriterAgent(BaseAgent):
    name = "Writer"
    model = MODEL_WRITER
    fallback_models = FALLBACK_WRITER
    temperature = TEMPERATURE_CREATIVE

    system_prompt = (
        "You are a talented fiction writer. "
        "You receive a JSON object describing a book chapter (title, plot, characters) "
        "and you must produce the FULL chapter text — vivid, engaging, and faithful "
        "to every detail in the description.\n\n"
        "RULES:\n"
        "1. Write ONLY the chapter text, starting with the chapter title as a heading.\n"
        "2. Include dialogue, descriptions, and inner thoughts where appropriate.\n"
        "3. Stay strictly within the plot and character list provided — do NOT invent "
        "new characters or plot points that are absent from the JSON.\n"
        "4. Aim for at least 800 words.\n"
        "5. Write in the same language as the JSON input (if the plot is in Italian, "
        "write in Italian; if in English, write in English)."
    )

    def run(self, *, chapter_json: dict, **kwargs: Any) -> str:
        """
        Generate the full chapter text from a chapter description.

        Args:
            chapter_json: dict with keys like "title", "plot", "characters".

        Returns:
            The generated chapter text (markdown-formatted).
        """
        # Present the JSON clearly to the model
        prompt = (
            "Here is the chapter description in JSON format:\n\n"
            f"```json\n{json.dumps(chapter_json, ensure_ascii=False, indent=2)}\n```\n\n"
            "Write the complete chapter now."
        )

        self.logger.info("Generating chapter: «%s»...", chapter_json.get("title", "?"))
        text = self._call_llm(prompt)
        self.logger.info("Chapter generated (%d chars).", len(text))
        return text

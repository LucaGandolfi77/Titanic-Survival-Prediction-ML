"""
Agent 2 — The Editor.

Receives the Writer's chapter text and the original JSON description.
Analyses the text for:
  • grammatical errors
  • logical inconsistencies
  • hallucinations (elements NOT present in the source JSON)

Returns a critical report AND a corrected version of the chapter.
"""

from __future__ import annotations

import json
from typing import Any

from config import MODEL_EDITOR, FALLBACK_EDITOR, TEMPERATURE_ANALYTICAL
from .base import BaseAgent


class EditorAgent(BaseAgent):
    name = "Editor"
    model = MODEL_EDITOR
    fallback_models = FALLBACK_EDITOR
    temperature = TEMPERATURE_ANALYTICAL

    system_prompt = (
        "You are a meticulous book editor and fact-checker.\n\n"
        "You will receive:\n"
        "  A) The ORIGINAL chapter description (JSON).\n"
        "  B) The DRAFT chapter text produced by a writer.\n\n"
        "Your job:\n"
        "1. Compare the draft against the JSON source of truth.\n"
        "2. List every issue you find, grouped under these headings:\n"
        "   ## Grammar & Style\n"
        "   ## Logical Inconsistencies\n"
        "   ## Hallucinations\n"
        "   (Hallucinations = characters, places, or events that appear in the "
        "draft but are NOT in the original JSON.)\n"
        "3. After the report, provide a section:\n"
        "   ## Corrected Chapter\n"
        "   containing the full corrected text.\n"
        "4. If the draft is already good, say so and return it unchanged under "
        "\"Corrected Chapter\".\n"
        "5. Keep the same language as the input."
    )

    def run(
        self,
        *,
        chapter_json: dict,
        draft_text: str,
        **kwargs: Any,
    ) -> str:
        """
        Edit and fact-check a chapter draft.

        Args:
            chapter_json: the original chapter description (source of truth).
            draft_text: the Writer's output.

        Returns:
            A markdown string with the editorial report + corrected chapter.
        """
        prompt = (
            "=== A) ORIGINAL CHAPTER DESCRIPTION (JSON) ===\n"
            f"```json\n{json.dumps(chapter_json, ensure_ascii=False, indent=2)}\n```\n\n"
            "=== B) DRAFT CHAPTER TEXT ===\n"
            f"{draft_text}\n\n"
            "Analyse the draft and provide your editorial report followed by "
            "the corrected chapter."
        )

        self.logger.info("Editing chapter: «%s»...", chapter_json.get("title", "?"))
        report = self._call_llm(prompt)
        self.logger.info("Editorial report ready (%d chars).", len(report))
        return report

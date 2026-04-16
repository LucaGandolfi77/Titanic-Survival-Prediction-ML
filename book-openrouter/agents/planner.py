"""
Agent 0 — The Planner.

Receives a high-level book description (title, genre, premise, characters)
and produces a structured outline.

Uses a TWO-PHASE approach to stay within free-model token limits:
  Phase A: generate a lightweight list of chapter titles + one-line synopses.
  Phase B: expand each chapter into a detailed outline (one API call each).
"""

from __future__ import annotations

import json
from typing import Any

from config import MODEL_PLANNER, FALLBACK_PLANNER, TEMPERATURE_ANALYTICAL
from .base import BaseAgent


def _clean_json(raw: str) -> str:
    """Strip markdown fences and leading/trailing noise from model output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    # Some models prepend text before the JSON — find the first [ or {
    for i, ch in enumerate(raw):
        if ch in ("[", "{"):
            raw = raw[i:]
            break
    return raw.strip()


class PlannerAgent(BaseAgent):
    name = "Planner"
    model = MODEL_PLANNER
    fallback_models = FALLBACK_PLANNER
    temperature = TEMPERATURE_ANALYTICAL
    system_prompt = ""  # built dynamically

    # ── Phase A: lightweight arc ────────────────────────────

    @staticmethod
    def _phase_a_system(n: int) -> str:
        return (
            "You are a master book architect.\n\n"
            f"Generate a list of EXACTLY {n} chapter titles with a ONE-LINE "
            "synopsis each.\n\n"
            "RULES:\n"
            "1. Respond ONLY with a valid JSON array.\n"
            "2. Each element: {\"chapter\": <number>, \"title\": \"...\", \"synopsis\": \"...\"}\n"
            "3. Build a coherent narrative arc (setup → rising action → "
            "complications → climax → resolution).\n"
            "4. Write in English.\n"
            "5. No markdown fences, no commentary — ONLY the JSON array."
        )

    # ── Phase B: expand one chapter ─────────────────────────

    @staticmethod
    def _phase_b_system() -> str:
        return (
            "You are a master book architect. You receive a book description "
            "and a chapter synopsis. Expand it into a DETAILED chapter outline.\n\n"
            "RULES:\n"
            "1. Respond ONLY with a valid JSON object.\n"
            "2. The object must have these keys:\n"
            '   {"chapter": <number>, "title": "...", "plot": "...", "characters": [...]}\n'
            "3. The 'plot' must be 4-6 detailed sentences.\n"
            "4. 'characters' is an array of {\"name\": \"...\", \"role\": \"...\", "
            "\"description\": \"...\"} — only characters appearing in THIS chapter.\n"
            "5. Write in English.\n"
            "6. No markdown fences, no commentary — ONLY the JSON object."
        )

    # ── Public interface ────────────────────────────────────

    def run(self, *, book_json: dict, num_chapters: int = 20, **kwargs: Any) -> list[dict]:
        """
        Generate chapter outlines in two phases for reliability.

        Phase A: one call → lightweight list of titles + synopses.
        Phase B: one call per chapter → detailed outline with characters.
        """
        title = book_json.get("title", "?")
        book_desc = json.dumps(book_json, ensure_ascii=False, indent=2)

        # ── Phase A ────────────────────────────────────────
        self.logger.info("Phase A — generating %d-chapter arc for «%s»...", num_chapters, title)
        self.system_prompt = self._phase_a_system(num_chapters)

        raw_a = self._call_llm(
            f"Book description:\n```json\n{book_desc}\n```\n\n"
            f"Generate the {num_chapters}-chapter arc. ONLY JSON array."
        )

        arc = self._parse_json_array(_clean_json(raw_a), "Phase A")
        self.logger.info("Phase A complete — %d chapter synopses.", len(arc))

        # ── Phase B ────────────────────────────────────────
        self.system_prompt = self._phase_b_system()
        detailed_chapters: list[dict] = []

        for item in arc:
            num = item.get("chapter", len(detailed_chapters) + 1)
            ch_title = item.get("title", f"Chapter {num}")
            synopsis = item.get("synopsis", "")

            self.logger.info(
                "Phase B — expanding chapter %d / %d: «%s»...",
                num, len(arc), ch_title,
            )

            raw_b = self._call_llm(
                f"Book description:\n```json\n{book_desc}\n```\n\n"
                f"Expand this chapter:\n"
                f"  Chapter {num}: \"{ch_title}\"\n"
                f"  Synopsis: {synopsis}\n\n"
                "Return ONLY the detailed JSON object for this chapter."
            )

            chapter = self._parse_json_object(_clean_json(raw_b), f"Phase B (ch.{num})")

            # Ensure required keys exist
            chapter.setdefault("chapter", num)
            chapter.setdefault("title", ch_title)
            chapter.setdefault("plot", synopsis)
            chapter.setdefault("characters", [])

            detailed_chapters.append(chapter)

        self.logger.info("Outline ready: %d chapters fully detailed.", len(detailed_chapters))
        return detailed_chapters

    # ── JSON parsing helpers ────────────────────────────────

    def _parse_json_array(self, raw: str, phase: str) -> list:
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.logger.error("Invalid JSON from %s:\n%s", phase, raw[:500])
            raise ValueError(
                f"Planner {phase}: invalid JSON. Try again or use a larger model."
            ) from exc
        if not isinstance(result, list):
            raise ValueError(f"Planner {phase}: expected JSON array, got {type(result).__name__}.")
        return result

    def _parse_json_object(self, raw: str, phase: str) -> dict:
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.logger.error("Invalid JSON from %s:\n%s", phase, raw[:500])
            raise ValueError(
                f"Planner {phase}: invalid JSON. Try again or use a larger model."
            ) from exc
        if not isinstance(result, dict):
            raise ValueError(f"Planner {phase}: expected JSON object, got {type(result).__name__}.")
        return result

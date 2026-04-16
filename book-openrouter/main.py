#!/usr/bin/env python3
"""
📚 Book Agents — Full-book multi-agent pipeline via OpenRouter (free models).

Usage:
    python main.py book.json
    python main.py book.json --output-dir ./my_book
    python main.py book.json --resume output/20260415_120000
    python main.py book.json --resume output/20260415_120000 --start-from 8
    python main.py book.json -v

Pipeline:
    Planner  → 20 chapter outlines
    For each chapter: Writer → Editor → Summarizer
    Translator → full book in Italian
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from agents import (  # noqa: E402
    PlannerAgent,
    WriterAgent,
    EditorAgent,
    SummarizerAgent,
    TranslatorAgent,
)


# ── Logging setup ───────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    """Configure console logging with a clean format."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s │ %(name)-12s │ %(levelname)-7s │ %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    for lib in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ── CLI argument parser ─────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="📚 Full-book pipeline + Italian translation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py book.json\n"
            "  python main.py book.json --chapters 10\n"
            "  python main.py book.json --resume output/20260415_120000\n"
            "  python main.py book.json --resume output/20260415_120000 --start-from 8\n"
        ),
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the JSON file with the book description.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (default: output/<timestamp>).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="DIR",
        help="Resume from an existing output directory (reuses outline + completed chapters).",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help="Start writing from chapter N (default: 1). Use with --resume.",
    )
    parser.add_argument(
        "--chapters",
        type=int,
        default=20,
        metavar="N",
        help="Number of chapters to generate (default: 20).",
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip the final Italian translation step.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


# ── Helper: extract corrected text from editorial report ────

def _extract_corrected(editorial_report: str, fallback: str) -> str:
    """Pull the '## Corrected Chapter' section, or return the fallback."""
    separator = "## Corrected Chapter"
    if separator in editorial_report:
        return editorial_report.split(separator, 1)[1].strip()
    return fallback


# ── Pipeline ────────────────────────────────────────────────

def run_pipeline(
    book_json: dict,
    output_dir: Path,
    *,
    num_chapters: int = 20,
    start_from: int = 1,
    skip_translation: bool = False,
) -> None:
    """
    Full-book pipeline:
      1. Planner    → N chapter outlines
      2. Per chapter: Writer → Editor → Summarizer
      3. Translator → full Italian book
    """
    logger = logging.getLogger("Pipeline")

    # ── Directory structure ─────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    chapters_dir = output_dir / "chapters_en"
    chapters_dir.mkdir(exist_ok=True)
    reports_dir = output_dir / "editorial_reports"
    reports_dir.mkdir(exist_ok=True)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    italian_dir = output_dir / "chapters_it"
    italian_dir.mkdir(exist_ok=True)

    outline_path = output_dir / "outline.json"
    book_en_path = output_dir / "full_book_en.md"
    book_it_path = output_dir / "full_book_it.md"

    # Save a copy of the input
    (output_dir / "input.json").write_text(
        json.dumps(book_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    book_title = book_json.get("title", "Untitled")

    # ═════════════════════════════════════════════════════════
    #  PHASE 1 — Planning (chapter outlines)
    # ═════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 1 — The Planner (%d-chapter outline)", num_chapters)
    logger.info("=" * 60)

    if outline_path.exists():
        logger.info("Outline already exists — loading from disk.")
        chapter_outlines = json.loads(outline_path.read_text(encoding="utf-8"))
    else:
        planner = PlannerAgent()
        chapter_outlines = planner.run(book_json=book_json, num_chapters=num_chapters)
        outline_path.write_text(
            json.dumps(chapter_outlines, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    total = len(chapter_outlines)
    logger.info("Book: «%s» — %d chapters planned.", book_title, total)

    # ═════════════════════════════════════════════════════════
    #  PHASE 2 — Per-chapter loop: Writer → Editor → Summarizer
    # ═════════════════════════════════════════════════════════
    writer = WriterAgent()
    editor = EditorAgent()
    summarizer = SummarizerAgent()

    # Load cumulative summary for continuity across chapters
    memory_path = output_dir / "memory.txt"
    memory = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""

    for outline in chapter_outlines:
        num = outline.get("chapter", chapter_outlines.index(outline) + 1)

        # Skip already-completed chapters when resuming
        if num < start_from:
            sum_path = summaries_dir / f"summary_{num:02d}.md"
            if sum_path.exists():
                memory = sum_path.read_text(encoding="utf-8")
            continue

        chapter_title = outline.get("title", f"Chapter {num}")

        logger.info("")
        logger.info("═" * 60)
        logger.info("CHAPTER %d / %d — «%s»", num, total, chapter_title)
        logger.info("═" * 60)

        # ── Writer ──────────────────────────────────────────
        logger.info("▸ Step 1/3 — Writer")

        # Inject previous-chapter memory so the Writer
        # maintains narrative continuity.
        chapter_json_with_memory = dict(outline)
        if memory:
            chapter_json_with_memory["previous_chapters_summary"] = memory

        draft = writer.run(chapter_json=chapter_json_with_memory)

        draft_path = chapters_dir / f"draft_{num:02d}.md"
        draft_path.write_text(
            f"# Chapter {num}: {chapter_title}\n\n{draft}",
            encoding="utf-8",
        )
        logger.info("  Draft saved → %s", draft_path.name)

        # ── Editor ──────────────────────────────────────────
        logger.info("▸ Step 2/3 — Editor")

        report = editor.run(chapter_json=outline, draft_text=draft)

        report_path = reports_dir / f"report_{num:02d}.md"
        report_path.write_text(
            f"# Editorial Report — Chapter {num}: {chapter_title}\n\n{report}",
            encoding="utf-8",
        )
        logger.info("  Report saved → %s", report_path.name)

        corrected = _extract_corrected(report, draft)
        final_path = chapters_dir / f"chapter_{num:02d}.md"
        final_path.write_text(
            f"# Chapter {num}: {chapter_title}\n\n{corrected}",
            encoding="utf-8",
        )
        logger.info("  Final chapter saved → %s", final_path.name)

        # ── Summarizer ──────────────────────────────────────
        logger.info("▸ Step 3/3 — Summarizer")

        summary = summarizer.run(chapter_text=corrected)

        sum_path = summaries_dir / f"summary_{num:02d}.md"
        sum_path.write_text(
            f"# Summary — Chapter {num}: {chapter_title}\n\n{summary}",
            encoding="utf-8",
        )
        logger.info("  Summary saved → %s", sum_path.name)

        # Update rolling memory for the next chapter's Writer
        memory = summary
        memory_path.write_text(memory, encoding="utf-8")

        logger.info("✓ Chapter %d complete.", num)

    # ═════════════════════════════════════════════════════════
    #  PHASE 3 — Assemble full English book
    # ═════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 3 — Assembling full English book")
    logger.info("=" * 60)

    full_en_parts = [f"# {book_title}\n"]
    for i in range(1, total + 1):
        chap_file = chapters_dir / f"chapter_{i:02d}.md"
        if chap_file.exists():
            full_en_parts.append(chap_file.read_text(encoding="utf-8"))
        else:
            full_en_parts.append(f"# Chapter {i}\n\n[NOT GENERATED]\n")
    full_en = "\n\n---\n\n".join(full_en_parts)
    book_en_path.write_text(full_en, encoding="utf-8")
    logger.info("Full English book saved → %s", book_en_path.name)

    # ═════════════════════════════════════════════════════════
    #  PHASE 4 — Translation to Italian (chapter by chapter)
    # ═════════════════════════════════════════════════════════
    if skip_translation:
        logger.info("Translation skipped (--skip-translation).")
    else:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 4 — The Translator (English → Italian)")
        logger.info("=" * 60)

        translator = TranslatorAgent()
        full_it_parts = [f"# {book_title} — Edizione Italiana\n"]

        for i in range(1, total + 1):
            chap_file = chapters_dir / f"chapter_{i:02d}.md"
            it_file = italian_dir / f"capitolo_{i:02d}.md"

            # Skip if already translated (for --resume)
            if it_file.exists():
                logger.info("  Chapter %d already translated, skipping.", i)
                full_it_parts.append(it_file.read_text(encoding="utf-8"))
                continue

            if not chap_file.exists():
                logger.warning("  Chapter %d not found, skipping translation.", i)
                continue

            en_text = chap_file.read_text(encoding="utf-8")
            it_text = translator.run(chapter_text=en_text, chapter_number=i)

            it_file.write_text(it_text, encoding="utf-8")
            full_it_parts.append(it_text)
            logger.info("  Chapter %d translated → %s", i, it_file.name)

        full_it = "\n\n---\n\n".join(full_it_parts)
        book_it_path.write_text(full_it, encoding="utf-8")
        logger.info("Full Italian book saved → %s", book_it_path.name)

    # ═════════════════════════════════════════════════════════
    #  Done
    # ═════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅  BOOK GENERATION COMPLETE")
    logger.info("   Title:    «%s»", book_title)
    logger.info("   Chapters: %d", total)
    logger.info("   Output:   %s", output_dir)
    logger.info("   English:  %s", book_en_path.name)
    if not skip_translation:
        logger.info("   Italian:  %s", book_it_path.name)
    logger.info("=" * 60)


# ── Entry point ─────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    logger = logging.getLogger("Main")

    # Validate API key
    config.validate()

    # Read input JSON
    json_path: Path = args.json_file.resolve()
    if not json_path.exists():
        logger.error("File not found: %s", json_path)
        sys.exit(1)

    try:
        book_json = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", json_path, exc)
        sys.exit(1)

    # Determine output directory
    if args.resume:
        output_dir = args.resume.resolve()
        logger.info("Resuming from %s", output_dir)
    elif args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.OUTPUT_DIR / timestamp

    logger.info("Book: «%s»", book_json.get("title", "?"))
    logger.info("Output: %s", output_dir)
    print()

    try:
        run_pipeline(
            book_json,
            output_dir,
            num_chapters=args.chapters,
            start_from=args.start_from,
            skip_translation=args.skip_translation,
        )
    except KeyboardInterrupt:
        print()
        logger.warning("Interrupted by user.")
        logger.info(
            "Resume with: python main.py %s --resume %s --start-from <N>",
            args.json_file, output_dir,
        )
        sys.exit(130)
    except (ConnectionError, RuntimeError) as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
book-to-speech — Convert PDF / EPUB / TXT books to audiobooks.

Pipeline:
  1. Read the input file (PDF, EPUB, TXT, MD)
  2. Optionally clean the text via a free OpenRouter LLM
  3. Split into chapters (if detectable)
  4. Synthesize each chapter to MP3 via edge-tts (free, no API key)

Usage:
  python main.py book.pdf
  python main.py book.epub --voice it-IT-DiegoNeural --skip-clean
  python main.py book.txt  --chapters --output-dir ./audiobook
  python main.py --list-voices it
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import config
from reader import read_file, split_into_chapters
from text_processor import clean_text
from tts_engine import synthesize, list_voices


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s │ %(name)-14s │ %(levelname)-7s │ %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def run_list_voices(language: str | None) -> None:
    """Print available voices and exit."""
    voices = asyncio.run(list_voices(language))
    if not voices:
        print(f"No voices found for language filter: '{language}'")
        sys.exit(1)

    print(f"\n{'Voice Name':<35} {'Gender':<10} {'Locale'}")
    print("─" * 65)
    for v in voices:
        print(f"{v['ShortName']:<35} {v['Gender']:<10} {v['Locale']}")
    print(f"\nTotal: {len(voices)} voices")


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    voice: str,
    rate: str,
    volume: str,
    skip_clean: bool,
    by_chapter: bool,
) -> None:
    """Main conversion pipeline."""
    logger = logging.getLogger("Pipeline")

    # 1. Read the book
    logger.info("═" * 60)
    logger.info("STEP 1 — Reading: %s", input_path.name)
    logger.info("═" * 60)
    raw_text = read_file(input_path)
    logger.info("Read %d characters from %s", len(raw_text), input_path.name)

    if not raw_text.strip():
        logger.error("The input file is empty or unreadable.")
        sys.exit(1)

    # 2. Clean text via LLM (optional)
    if skip_clean:
        logger.info("Skipping LLM text cleaning (--skip-clean).")
        text = raw_text
    else:
        logger.info("═" * 60)
        logger.info("STEP 2 — Cleaning text via LLM")
        logger.info("═" * 60)
        config.validate()
        text = clean_text(raw_text)
        # Save cleaned text for inspection
        cleaned_path = output_dir / "cleaned_text.txt"
        cleaned_path.write_text(text, encoding="utf-8")
        logger.info("Cleaned text saved: %s", cleaned_path)

    # 3. Split & synthesize
    logger.info("═" * 60)
    logger.info("STEP 3 — Synthesizing audio (voice=%s)", voice)
    logger.info("═" * 60)

    if by_chapter:
        chapters = split_into_chapters(text)
        logger.info("Detected %d chapters.", len(chapters))

        for i, chapter_text in enumerate(chapters, 1):
            out_file = output_dir / f"chapter_{i:02d}.mp3"
            logger.info("─ Chapter %d / %d (%d chars)", i, len(chapters), len(chapter_text))
            synthesize(chapter_text, out_file, voice=voice, rate=rate, volume=volume)
    else:
        out_file = output_dir / f"{input_path.stem}.mp3"
        synthesize(text, out_file, voice=voice, rate=rate, volume=volume)

    logger.info("═" * 60)
    logger.info("✓ DONE — Audio files saved in: %s", output_dir)
    logger.info("═" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="📖🔊 Book-to-Speech — Convert books to audiobooks for free.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py book.pdf\n"
            "  python main.py book.epub --voice it-IT-DiegoNeural\n"
            "  python main.py book.txt  --chapters --skip-clean\n"
            "  python main.py --list-voices it\n"
        ),
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to a PDF, EPUB, TXT, or MD file.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (default: output/<timestamp>).",
    )
    parser.add_argument(
        "--voice",
        default=config.TTS_VOICE,
        help=f"TTS voice name (default: {config.TTS_VOICE}). "
             "Use --list-voices to see all options.",
    )
    parser.add_argument(
        "--rate",
        default=config.TTS_RATE,
        help=f"Speech rate adjustment (default: '{config.TTS_RATE}'). "
             "E.g. '+15%%' or '-10%%'.",
    )
    parser.add_argument(
        "--volume",
        default=config.TTS_VOLUME,
        help=f"Volume adjustment (default: '{config.TTS_VOLUME}').",
    )
    parser.add_argument(
        "--chapters",
        action="store_true",
        help="Split into separate audio files per chapter.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip LLM text cleaning (use raw extracted text).",
    )
    parser.add_argument(
        "--list-voices",
        nargs="?",
        const="",
        metavar="LANG",
        help="List available voices and exit. Optionally filter by language "
             "(e.g. 'it', 'en', 'de').",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Handle --list-voices
    if args.list_voices is not None:
        lang = args.list_voices if args.list_voices else None
        run_list_voices(lang)
        return

    # Require input file for normal operation
    if not args.input_file:
        parser.error("Please provide an input file (or use --list-voices).")

    input_path = Path(args.input_file).resolve()
    if not input_path.is_file():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.OUTPUT_DIR / f"{input_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Main")
    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_dir)
    logger.info("Voice:  %s", args.voice)

    run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        voice=args.voice,
        rate=args.rate,
        volume=args.volume,
        skip_clean=args.skip_clean,
        by_chapter=args.chapters,
    )


if __name__ == "__main__":
    main()

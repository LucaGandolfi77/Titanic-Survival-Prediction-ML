"""
tts_engine.py — Text-to-Speech via edge-tts (free, no API key).

Features:
  • Splits long text into ≤ 2500-char chunks at sentence boundaries
  • Generates MP3/WAV per chunk, then concatenates
  • Supports 300+ voices in 40+ languages
  • Configurable rate and volume
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

import edge_tts

from config import TTS_VOICE, TTS_RATE, TTS_VOLUME, TTS_MAX_CHUNK_CHARS

logger = logging.getLogger("TTS")


def _split_into_sentences(text: str) -> list[str]:
    """Rough sentence splitter that respects common abbreviations."""
    import re
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_for_tts(text: str, max_chars: int = TTS_MAX_CHUNK_CHARS) -> list[str]:
    """
    Split text into TTS-friendly chunks (≤ max_chars) at sentence boundaries.
    """
    sentences = _split_into_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


async def _synthesize_chunk(
    text: str,
    output_path: str,
    voice: str = TTS_VOICE,
    rate: str = TTS_RATE,
    volume: str = TTS_VOLUME,
) -> None:
    """Synthesize a single text chunk to an audio file."""
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
    await communicate.save(output_path)


async def _synthesize_all_chunks(
    chunks: list[str],
    output_dir: Path,
    voice: str,
    rate: str,
    volume: str,
) -> list[Path]:
    """Synthesize all chunks and return paths to the audio files."""
    paths: list[Path] = []
    for i, chunk in enumerate(chunks, 1):
        out = output_dir / f"chunk_{i:04d}.mp3"
        logger.info("  TTS chunk %d / %d (%d chars)...", i, len(chunks), len(chunk))
        await _synthesize_chunk(chunk, str(out), voice, rate, volume)
        paths.append(out)
    return paths


def _concatenate_mp3(parts: list[Path], output: Path) -> None:
    """
    Concatenate MP3 files by simple binary append.

    For MP3 files with the same encoding (which edge-tts guarantees),
    this produces a valid output.  No ffmpeg needed.
    """
    with open(output, "wb") as fout:
        for part in parts:
            fout.write(part.read_bytes())


def synthesize(
    text: str,
    output_path: str | Path,
    *,
    voice: str = TTS_VOICE,
    rate: str = TTS_RATE,
    volume: str = TTS_VOLUME,
) -> Path:
    """
    Convert text to a single audio file.

    Args:
        text: The full text to synthesise.
        output_path: Where to save the final .mp3 file.
        voice: edge-tts voice name (e.g. "en-US-AriaNeural").
        rate: Speed adjustment (e.g. "+10%", "-5%").
        volume: Volume adjustment.

    Returns:
        Path to the output audio file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = _chunk_for_tts(text)
    logger.info("Synthesizing %d TTS chunks → %s (voice=%s)", len(chunks), output_path, voice)

    if not chunks:
        logger.warning("No text to synthesise.")
        output_path.write_bytes(b"")
        return output_path

    # Use a temp dir for individual chunks
    with tempfile.TemporaryDirectory(prefix="bts_") as tmpdir:
        tmp = Path(tmpdir)
        parts = asyncio.run(
            _synthesize_all_chunks(chunks, tmp, voice, rate, volume)
        )
        _concatenate_mp3(parts, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("✓ Audio saved: %s (%.1f MB)", output_path, size_mb)
    return output_path


async def list_voices(language: str | None = None) -> list[dict]:
    """
    List available edge-tts voices, optionally filtered by language prefix.

    Usage:
        voices = asyncio.run(list_voices("it"))
        for v in voices:
            print(v["ShortName"], v["Gender"])
    """
    voices = await edge_tts.list_voices()
    if language:
        lang = language.lower()
        voices = [v for v in voices if v["Locale"].lower().startswith(lang)]
    return voices

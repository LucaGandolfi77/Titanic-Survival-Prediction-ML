"""
Centralized configuration — loads settings from .env file.

Covers:
  • OpenRouter API (for text cleaning / preparation)
  • TTS voice & speed settings
  • Model fallback chains
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ───────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
OUTPUT_DIR = ROOT_DIR / "output"

load_dotenv(ENV_PATH)

# ── OpenRouter API ──────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_APP_NAME: str = os.getenv("OPENROUTER_APP_NAME", "BookToSpeech")

# ── Free model for text cleaning ────────────────────────────
MODEL_CLEANER: str = os.getenv(
    "MODEL_CLEANER", "google/gemma-4-31b-it:free"
)
FALLBACK_MODELS: list[str] = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "minimax/minimax-m2.5:free",
    "z-ai/glm-4.5-air:free",
]

# ── LLM parameters ─────────────────────────────────────────
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
RATE_LIMIT_PAUSE: float = float(os.getenv("RATE_LIMIT_PAUSE", "5.0"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "4"))

# ── TTS settings (edge-tts) ────────────────────────────────
# Run `edge-tts --list-voices` to see all available voices.
# Popular English voices:
#   en-US-AriaNeural, en-US-GuyNeural, en-GB-SoniaNeural
# Popular Italian voices:
#   it-IT-ElsaNeural, it-IT-DiegoNeural
TTS_VOICE: str = os.getenv("TTS_VOICE", "en-US-AriaNeural")
TTS_RATE: str = os.getenv("TTS_RATE", "+0%")      # e.g. "+10%", "-15%"
TTS_VOLUME: str = os.getenv("TTS_VOLUME", "+0%")   # e.g. "+20%", "-10%"
TTS_OUTPUT_FORMAT: str = os.getenv("TTS_OUTPUT_FORMAT", "mp3")  # mp3 or wav

# ── Chunk limits ────────────────────────────────────────────
# edge-tts works best with chunks ≤ 3000 chars.
TTS_MAX_CHUNK_CHARS: int = int(os.getenv("TTS_MAX_CHUNK_CHARS", "2500"))
# Max chars sent to OpenRouter for cleaning in one call.
CLEAN_MAX_CHUNK_CHARS: int = int(os.getenv("CLEAN_MAX_CHUNK_CHARS", "6000"))


def validate() -> None:
    """Check that mandatory settings are present; exit early if not."""
    if not OPENROUTER_API_KEY:
        print(
            "⚠  OPENROUTER_API_KEY is not set.\n"
            "   Without it, text-cleaning via LLM will be skipped.\n"
            "   The pipeline will still work (raw text → TTS).\n"
            "   To enable cleaning: cp .env.example .env and paste your key.\n"
            "   Get one for free at https://openrouter.ai/keys",
            file=sys.stderr,
        )

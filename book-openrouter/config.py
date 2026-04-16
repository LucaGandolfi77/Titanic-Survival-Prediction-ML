"""
Centralized configuration — loads settings from .env file.

All OpenRouter-related constants (API key, base URL, model names)
are defined here so that agents never hardcode credentials.
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

# Optional: some OpenRouter endpoints honour these headers for
# analytics / priority routing.
OPENROUTER_APP_NAME: str = os.getenv("OPENROUTER_APP_NAME", "BookAgents")
OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "")

# ── Free model assignments ──────────────────────────────────
# Each agent gets its own model so you can swap them independently.
# Only ":free" suffixed models are used — zero cost.
MODEL_WRITER: str = os.getenv(
    "MODEL_WRITER", "nvidia/nemotron-3-super-120b-a12b:free"
)
MODEL_EDITOR: str = os.getenv(
    "MODEL_EDITOR", "google/gemma-4-31b-it:free"
)
MODEL_SUMMARIZER: str = os.getenv(
    "MODEL_SUMMARIZER", "openai/gpt-oss-120b:free"
)
MODEL_PLANNER: str = os.getenv(
    "MODEL_PLANNER", "nvidia/nemotron-3-super-120b-a12b:free"
)
MODEL_TRANSLATOR: str = os.getenv(
    "MODEL_TRANSLATOR", "google/gemma-4-31b-it:free"
)

# ── Fallback model chains ───────────────────────────────────
# If the primary model fails (404, persistent 429, etc.) the agent
# automatically tries the next model in the list.  Order = preference.
FALLBACK_WRITER: list[str] = [
    "openai/gpt-oss-120b:free",
    "arcee-ai/trinity-large-preview:free",
    "minimax/minimax-m2.5:free",
    "z-ai/glm-4.5-air:free",
]
FALLBACK_EDITOR: list[str] = [
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "z-ai/glm-4.5-air:free",
]
FALLBACK_SUMMARIZER: list[str] = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "z-ai/glm-4.5-air:free",
    "minimax/minimax-m2.5:free",
]
FALLBACK_PLANNER: list[str] = [
    "openai/gpt-oss-120b:free",
    "arcee-ai/trinity-large-preview:free",
    "minimax/minimax-m2.5:free",
    "z-ai/glm-4.5-air:free",
]
FALLBACK_TRANSLATOR: list[str] = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "openai/gpt-oss-120b:free",
    "z-ai/glm-4.5-air:free",
]

# ── Generation parameters ───────────────────────────────────
TEMPERATURE_CREATIVE: float = float(os.getenv("TEMPERATURE_CREATIVE", "0.85"))
TEMPERATURE_ANALYTICAL: float = float(os.getenv("TEMPERATURE_ANALYTICAL", "0.3"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))

# ── Rate-limiting ───────────────────────────────────────────
# Pause (seconds) inserted between consecutive API calls to
# stay within the free-tier rate limits.
RATE_LIMIT_PAUSE: float = float(os.getenv("RATE_LIMIT_PAUSE", "5.0"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "5"))


def validate() -> None:
    """Check that mandatory settings are present; exit early if not."""
    if not OPENROUTER_API_KEY:
        print(
            "ERROR: OPENROUTER_API_KEY is not set.\n"
            "Copy .env.example to .env and paste your key.\n"
            "Get one for free at https://openrouter.ai/keys",
            file=sys.stderr,
        )
        sys.exit(1)

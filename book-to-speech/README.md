# 📖🔊 Book-to-Speech

Convert **PDF**, **EPUB**, or **TXT** books into audiobooks — completely free.

## How it works

```
┌──────────┐     ┌──────────────┐     ┌───────────┐     ┌───────────┐
│ PDF/EPUB │ ──▶ │  Text Reader │ ──▶ │ LLM Clean │ ──▶ │ edge-tts  │ ──▶ 🔊 MP3
│  / TXT   │     │  (pypdf,     │     │ (optional, │     │ (free,    │
│          │     │   ebooklib)  │     │  OpenRouter│     │  no key)  │
└──────────┘     └──────────────┘     │  free tier)│     └───────────┘
                                      └───────────┘
```

| Component | Cost | What it does |
|-----------|------|--------------|
| **Text Reader** | Free | Extracts text from PDF/EPUB/TXT |
| **LLM Cleaner** | Free (OpenRouter) | Removes OCR noise, normalises text for narration |
| **TTS Engine** | Free (edge-tts) | Microsoft Edge TTS — 300+ voices, 40+ languages |

## Quick start

```bash
# 1. Install dependencies
cd book-to-speech
pip install -r requirements.txt

# 2. (Optional) Set up OpenRouter for LLM text cleaning
cp .env.example .env
# Edit .env and paste your free key from https://openrouter.ai/keys

# 3. Convert a book!
python main.py book.pdf                          # basic conversion
python main.py book.epub --chapters              # split by chapter
python main.py book.txt --skip-clean             # skip LLM cleaning
python main.py book.pdf --voice it-IT-DiegoNeural  # Italian voice
```

## Usage

```
python main.py [OPTIONS] INPUT_FILE

positional arguments:
  INPUT_FILE            Path to a PDF, EPUB, TXT, or MD file.

options:
  -o, --output-dir DIR  Output directory (default: output/<timestamp>)
  --voice VOICE         TTS voice (default: en-US-AriaNeural)
  --rate RATE           Speed: "+15%", "-10%" (default: "+0%")
  --volume VOL          Volume: "+20%", "-5%" (default: "+0%")
  --chapters            Split into separate MP3s per chapter
  --skip-clean          Skip LLM text cleaning (raw text → audio)
  --list-voices [LANG]  List available voices and exit
  -v, --verbose         Debug logging
```

## List voices

```bash
python main.py --list-voices          # all 300+ voices
python main.py --list-voices en       # English only
python main.py --list-voices it       # Italian only
python main.py --list-voices de       # German only
```

## Popular voices

| Voice | Language | Gender |
|-------|----------|--------|
| `en-US-AriaNeural` | English (US) | Female |
| `en-US-GuyNeural` | English (US) | Male |
| `en-GB-SoniaNeural` | English (UK) | Female |
| `it-IT-ElsaNeural` | Italian | Female |
| `it-IT-DiegoNeural` | Italian | Male |
| `de-DE-KatjaNeural` | German | Female |
| `fr-FR-DeniseNeural` | French | Female |
| `es-ES-ElviraNeural` | Spanish | Female |
| `ja-JP-NanamiNeural` | Japanese | Female |

## Project structure

```
book-to-speech/
├── main.py              # CLI entry point
├── config.py            # Settings (API key, voice, models)
├── reader.py            # PDF / EPUB / TXT reader
├── text_processor.py    # LLM text cleaning (OpenRouter free)
├── tts_engine.py        # edge-tts audio synthesis
├── requirements.txt     # Python dependencies
├── .env.example         # Configuration template
└── output/              # Generated audiobooks
```

## Without an API key

The pipeline works perfectly **without** an OpenRouter API key — it just
skips the LLM text-cleaning step and sends the raw extracted text
directly to TTS.  For clean source files (well-formatted TXT, EPUB),
this is often good enough.

## Dependencies

- **edge-tts** — Microsoft Edge's TTS engine. Free, high quality, no
  API key required, 300+ voices.
- **pypdf** — PDF text extraction.
- **ebooklib** + **beautifulsoup4** — EPUB parsing.
- **openai** — OpenRouter client for the optional LLM cleaning step.

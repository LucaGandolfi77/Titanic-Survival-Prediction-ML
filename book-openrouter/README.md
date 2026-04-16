# 📚 Book Agents — Full-Book Pipeline via OpenRouter (Free Models)

A Python system that coordinates **5 specialized agents** to plan, write, edit, summarize, and translate a complete **20-chapter book** — powered entirely by **free models** on [OpenRouter](https://openrouter.ai).

```
book.json
    │
    ▼
┌──────────┐
│ Planner  │ → outline.json (20 chapters)
└────┬─────┘
     │
     ▼  ×20 chapters
  ┌────────────────────────────────────┐
  │  Writer  → draft_XX.md            │
  │  Editor  → report_XX.md           │
  │           → chapter_XX.md (final) │
  │  Summarizer → summary_XX.md       │
  └───────────────┬────────────────────┘
                  │
                  ▼
           full_book_en.md
                  │
                  ▼
           ┌────────────┐
           │ Translator │ ×20 chapters
           └─────┬──────┘
                 ▼
           full_book_it.md  🇮🇹
```

## Agents

| Agent | Task | Default Free Model |
|---|---|---|
| **Planner** | Generates 20-chapter outline from book description | `nvidia/nemotron-3-super-120b-a12b:free` |
| **Writer** | Writes each chapter from its outline | `nvidia/nemotron-3-super-120b-a12b:free` |
| **Editor** | Grammar, logic, hallucination check + corrected version | `google/gemma-4-31b-it:free` |
| **Summarizer** | Executive summary per chapter (used as memory for next) | `openai/gpt-oss-120b:free` |
| **Translator** | Translates each chapter English → Italian | `google/gemma-4-31b-it:free` |

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
cd openrouter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
```

Open `.env` and paste your OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

> **Free key:** sign up at [openrouter.ai/keys](https://openrouter.ai/keys) — no credit card required.

### 3. Run the pipeline

```bash
# Generate a full 20-chapter book + Italian translation
python main.py example_input.json

# Verbose mode
python main.py example_input.json -v

# Custom output folder
python main.py example_input.json --output-dir ./my_book

# Resume after interruption (skips completed chapters)
python main.py example_input.json --resume output/20260415_120000

# Resume from a specific chapter
python main.py example_input.json --resume output/20260415_120000 --start-from 8

# Write only (skip Italian translation)
python main.py example_input.json --skip-translation
```

---

## 📄 Input JSON Format

The input is a **book-level description** (not a single chapter):

```json
{
  "title": "Book Title",
  "genre": "Genre",
  "premise": "Full premise / synopsis of the book...",
  "characters": [
    {
      "name": "Character Name",
      "role": "protagonist / antagonist / etc.",
      "description": "Character description."
    }
  ],
  "setting": "Where and when the story takes place.",
  "tone": "Narrative tone and style."
}
```

See [example_input.json](example_input.json) for a complete example.

---

## 📁 Output Structure

```
output/20260415_120000/
├── input.json                    # copy of book description
├── outline.json                  # 20-chapter plan (from Planner)
├── memory.txt                    # rolling summary for continuity
├── chapters_en/
│   ├── draft_01.md … draft_20.md       # raw Writer output
│   └── chapter_01.md … chapter_20.md   # final (post-Editor)
├── editorial_reports/
│   └── report_01.md … report_20.md     # Editor analysis
├── summaries/
│   └── summary_01.md … summary_20.md   # per-chapter summaries
├── chapters_it/
│   └── capitolo_01.md … capitolo_20.md # Italian translations
├── full_book_en.md               # complete English book
└── full_book_it.md               # complete Italian book 🇮🇹
```

---

## 🏗️ Code Architecture

```
openrouter/
├── main.py              # CLI entry point & full-book pipeline
├── config.py            # .env loading, constants, validation
├── agents/
│   ├── base.py          # BaseAgent: OpenAI client, retry, rate-limit
│   ├── planner.py       # Agent 0 — The Planner (20-chapter outline)
│   ├── writer.py        # Agent 1 — The Writer
│   ├── editor.py        # Agent 2 — The Editor
│   ├── summarizer.py    # Agent 3 — The Summarizer
│   └── translator.py    # Agent 4 — The Translator (EN → IT)
├── requirements.txt
├── .env.example
├── example_input.json   # ready-to-use book description
└── README.md
```

### Design Decisions

- **Planner-first architecture** — a dedicated agent generates the full 20-chapter arc before writing begins, ensuring coherent story structure.
- **Rolling memory** — each chapter's summary is fed to the next Writer call, maintaining narrative continuity without exceeding context windows.
- **Chapter-by-chapter translation** — translating one chapter at a time avoids token limits and allows resuming after interruption.
- **Full resumability** — outline, chapters, and translations are saved individually; `--resume` + `--start-from` lets you continue from any point.

---

## ⚙️ Configuring Free Models

All model assignments live in `.env`. You can swap them with any model ending in `:free` from [openrouter.ai/models](https://openrouter.ai/models?q=free).

Popular free options:

| Model | Strengths | Good For |
|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b:free` | 120B MoE, strong creative + reasoning | Writer |
| `google/gemma-4-31b-it:free` | Precise, analytical, 256K context | Editor |
| `openai/gpt-oss-120b:free` | 120B MoE, structured output | Summarizer |
| `minimax/minimax-m2.5:free` | Productive, good at documents | Any role |
| `z-ai/glm-4.5-air:free` | Hybrid reasoning/non-reasoning | Any role |

> **Tip:** model availability on OpenRouter changes. If a model returns errors, check the [models page](https://openrouter.ai/models) and swap to another `:free` one.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `OPENROUTER_API_KEY is not set` | Copy `.env.example` → `.env` and add your key |
| 429 errors / rate limiting | Increase `RATE_LIMIT_PAUSE` in `.env` (e.g. `5.0`) |
| Empty or truncated output | Increase `MAX_TOKENS` in `.env` |
| Model not available | Check [openrouter.ai/models](https://openrouter.ai/models) and update the model name in `.env` |
| `JSONDecodeError` on input | Validate your JSON with `python -m json.tool chapter.json` |

from pathlib import Path

APP_NAME    = "Local AI Runner"
APP_VERSION = "1.0.0"

MODELS_DIR = Path.home() / ".local_ai_runner" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CTX       = 4096
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMP      = 0.7
DEFAULT_TOP_P     = 0.95

MODEL_CATALOG = [            # ── Ultra-lightweight (< 1B) ─────────────────────────────────────────────

    {
        "id":       "qwen2.5-0.5b-q4",
        "name":     "Qwen 2.5 0.5B Instruct",
        "author":   "Alibaba",
        "filename": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF"
                    "/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "size_gb":  0.40,
        "ram_gb":   1,
        "context":  32768,
        "desc":     "Smallest usable model. ~400MB. Fits on any device including Raspberry Pi.",
        "tags":     ["ultra-light", "edge", "fast"],
    },

    # ── Tiny (1B–2B) ──────────────────────────────────────────────────────────

    # Add/replace in MODEL_CATALOG in config.py

# ── 135M ─────────────────────────────────────────────────────────────────────
{
    "id":       "qwen2.5-0.5b-q4",
    "name":     "Qwen 2.5 0.5B Instruct",
    "author":   "Alibaba",
    "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "repo_id":  "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "url":      "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
                "/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "size_gb":  0.40, "ram_gb": 1, "context": 32768,
    "desc":     "Smallest usable LLM. Fits on any device including Raspberry Pi.",
    "tags":     ["ultra-light", "edge", "fast"],
},

# ── 0.6B ─────────────────────────────────────────────────────────────────────
{
    "id":       "smollm2-360m-q4",
    "name":     "SmolLM2 360M Instruct",
    "author":   "HuggingFace",
    "filename": "SmolLM2-360M-Instruct-Q4_K_M.gguf",
    "repo_id":  "bartowski/SmolLM2-360M-Instruct-GGUF",
    "url":      "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF"
                "/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
    "size_gb":  0.25, "ram_gb": 1, "context": 8192,
    "desc":     "HuggingFace 360M. Extremely fast. Good for simple completions.",
    "tags":     ["ultra-light", "edge", "fast"],
},

# ── 1.1B ─────────────────────────────────────────────────────────────────────
{
    "id":       "smollm2-1.7b-q4",
    "name":     "SmolLM2 1.7B Instruct",
    "author":   "HuggingFace",
    "filename": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    "repo_id":  "bartowski/SmolLM2-1.7B-Instruct-GGUF",
    "url":      "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF"
                "/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    "size_gb":  1.05, "ram_gb": 2, "context": 8192,
    "desc":     "Excellent reasoning above its size. HuggingFace's own model.",
    "tags":     ["ultra-light", "reasoning"],
},

# ── 1.5B ─────────────────────────────────────────────────────────────────────
{
    "id":       "tinyllama-1.1b-q4",
    "name":     "TinyLlama 1.1B Chat v1.0",
    "author":   "Zhang Peiyuan",
    "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "repo_id":  "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "url":      "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
                "/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "size_gb":  0.67, "ram_gb": 2, "context": 2048,
    "desc":     "Classic reliable tiny model. Trained on 3T tokens. Good for CI/pipelines.",
    "tags":     ["ultra-light", "classic", "fast"],
},

# ── 1.5B (Qwen official repo — more stable than bartowski for this size) ──────
{
    "id":       "qwen2.5-1.5b-q4",
    "name":     "Qwen 2.5 1.5B Instruct",
    "author":   "Alibaba",
    "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "repo_id":  "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "url":      "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF"
                "/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "size_gb":  0.99, "ram_gb": 2, "context": 32768,
    "desc":     "Excellent coding at 1.5B. Official Alibaba repo — very stable.",
    "tags":     ["fast", "multilingual", "long-context", "code"],
},

# ── 1B (Gemma 3 QAT — better quality/size than standard 4-bit) ───────────────
{
    "id":       "gemma-3-1b-qat",
    "name":     "Gemma 3 1B Instruct QAT",
    "author":   "Google",
    "filename": "gemma-3-1b-it-qat-q4_0.gguf",
    "repo_id":  "google/gemma-3-1b-it-qat-q4_0-gguf",
    "url":      "https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf"
                "/resolve/main/gemma-3-1b-it-qat-q4_0.gguf",
    "size_gb":  0.54, "ram_gb": 2, "context": 32768,
    "desc":     "Google's QAT model: 4-bit quality close to bfloat16. Official Google repo.",
    "tags":     ["ultra-light", "quality", "qat"],
},

    {
        "id":       "tinyllama-1.1b-q4",
        "name":     "TinyLlama 1.1B Chat v1.0",
        "author":   "Zhang Peiyuan",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "url":      "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
                    "/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb":  0.67,
        "ram_gb":   2,
        "context":  2048,
        "desc":     "Classic tiny chat model. Very fast on CPU. Good for embedded / CI pipelines.",
        "tags":     ["ultra-light", "fast", "classic"],
    },
    {
        "id":       "llama-3.2-1b-q4",
        "name":     "Llama 3.2 1B Instruct",
        "author":   "Meta",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
                    "/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb":  0.81,
        "ram_gb":   2,
        "context":  8192,
        "desc":     "Meta's modern 1B model. Much better instruction-following than TinyLlama.",
        "tags":     ["fast", "lightweight"],
    },
    {
        "id":       "smollm2-1.7b-q4",
        "name":     "SmolLM2 1.7B Instruct",
        "author":   "HuggingFace",
        "filename": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF"
                    "/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "size_gb":  1.05,
        "ram_gb":   2,
        "context":  8192,
        "desc":     "HuggingFace's own compact model. Punches above its weight on reasoning tasks.",
        "tags":     ["ultra-light", "reasoning", "fast"],
    },
    {
        "id":       "qwen2.5-1.5b-q4",
        "name":     "Qwen 2.5 1.5B Instruct",
        "author":   "Alibaba",
        "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF"
                    "/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "size_gb":  0.99,
        "ram_gb":   2,
        "context":  32768,
        "desc":     "Tiny yet capable. Very long context (32K). Great multilingual support.",
        "tags":     ["fast", "multilingual", "long-context"],
    },
    {
        "id":       "gemma-2-2b-q4",
        "name":     "Gemma 2 2B Instruct",
        "author":   "Google",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF"
                    "/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "size_gb":  1.63,
        "ram_gb":   4,
        "context":  8192,
        "desc":     "Google's efficient small model. Solid general-purpose quality.",
        "tags":     ["balanced", "quality"],
    },

    # ── Small (3B–4B) ─────────────────────────────────────────────────────────

    {
        "id":       "llama-3.2-3b-q4",
        "name":     "Llama 3.2 3B Instruct",
        "author":   "Meta",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
                    "/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb":  2.02,
        "ram_gb":   4,
        "context":  8192,
        "desc":     "Best-in-class 3B. Beats Gemma 2 2B and Phi-3.5 Mini on most tasks.",
        "tags":     ["balanced"],
    },
    {
        "id":       "qwen2.5-3b-q4",
        "name":     "Qwen 2.5 3B Instruct",
        "author":   "Alibaba",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF"
                    "/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "size_gb":  1.94,
        "ram_gb":   4,
        "context":  32768,
        "desc":     "Qwen's 3B. Excellent coding and long-context at 4GB RAM.",
        "tags":     ["balanced", "code", "long-context"],
    },
    {
        "id":       "phi-3.5-mini-q4",
        "name":     "Phi 3.5 Mini Instruct",
        "author":   "Microsoft",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF"
                    "/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb":  2.39,
        "ram_gb":   4,
        "context":  128000,
        "desc":     "Massive 128K context window. Best for long documents and RAG pipelines.",
        "tags":     ["reasoning", "long-context"],
    },

    # ── Medium (7B–9B) ────────────────────────────────────────────────────────

    {
        "id":       "qwen2.5-7b-q4",
        "name":     "Qwen 2.5 7B Instruct",
        "author":   "Alibaba",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF"
                    "/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb":  4.68,
        "ram_gb":   8,
        "context":  32768,
        "desc":     "Top open-source 7B. Excellent reasoning, code and multilingual. Needs 8GB RAM.",
        "tags":     ["powerful", "multilingual", "code"],
    },
    {
        "id":       "mistral-7b-v0.3-q4",
        "name":     "Mistral 7B Instruct v0.3",
        "author":   "Mistral AI",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF"
                    "/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size_gb":  4.37,
        "ram_gb":   8,
        "context":  32768,
        "desc":     "Classic 7B with function calling and structured output. Great for agents.",
        "tags":     ["powerful", "creative", "function-calling"],
    },
    {
        "id":       "llama-3.1-8b-q4",
        "name":     "Llama 3.1 8B Instruct",
        "author":   "Meta",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
                    "/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb":  4.92,
        "ram_gb":   8,
        "context":  131072,
        "desc":     "Meta's best 8B with 128K context. Strong all-rounder, great for chat and code.",
        "tags":     ["powerful", "long-context", "code"],
    },
    {
        "id":       "gemma-2-9b-q4",
        "name":     "Gemma 2 9B Instruct",
        "author":   "Google",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF"
                    "/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
        "size_gb":  5.77,
        "ram_gb":   10,
        "context":  8192,
        "desc":     "Google's 9B powerhouse. High quality text generation, needs ~10GB RAM.",
        "tags":     ["powerful", "quality"],
    },
    {

        "id":       "llama-3.2-1b-q4",
        "name":     "Llama 3.2 1B Instruct",
        "author":   "Meta",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
                    "/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb":  0.81,
        "ram_gb":   2,
        "context":  8192,
        "desc":     "Very fast. Ideal for low-end hardware and quick tasks.",
        "tags":     ["fast", "lightweight"],
    },
    {
        "id":       "llama-3.2-3b-q4",
        "name":     "Llama 3.2 3B Instruct",
        "author":   "Meta",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
                    "/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb":  2.02,
        "ram_gb":   4,
        "context":  8192,
        "desc":     "Balanced quality for most tasks. Runs on 4 GB RAM.",
        "tags":     ["balanced"],
    },
    {
        "id":       "qwen2.5-1.5b-q4",
        "name":     "Qwen 2.5 1.5B Instruct",
        "author":   "Alibaba",
        "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF"
                    "/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "size_gb":  0.99,
        "ram_gb":   2,
        "context":  32768,
        "desc":     "Tiny yet capable. Very long context. Multilingual.",
        "tags":     ["fast", "multilingual", "long-context"],
    },
    {
        "id":       "qwen2.5-7b-q4",
        "name":     "Qwen 2.5 7B Instruct",
        "author":   "Alibaba",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF"
                    "/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb":  4.68,
        "ram_gb":   8,
        "context":  32768,
        "desc":     "High quality reasoning and code. Needs 8 GB RAM.",
        "tags":     ["powerful", "multilingual", "code"],
    },
    {
        "id":       "phi-3.5-mini-q4",
        "name":     "Phi 3.5 Mini Instruct",
        "author":   "Microsoft",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF"
                    "/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb":  2.39,
        "ram_gb":   4,
        "context":  128000,
        "desc":     "Excellent reasoning for its size. Huge context window.",
        "tags":     ["reasoning", "long-context"],
    },
    {
        "id":       "gemma-2-2b-q4",
        "name":     "Gemma 2 2B Instruct",
        "author":   "Google",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF"
                    "/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "size_gb":  1.63,
        "ram_gb":   4,
        "context":  8192,
        "desc":     "Google's efficient small model. Good general purpose.",
        "tags":     ["balanced", "quality"],
    },
    {
        "id":       "mistral-7b-v0.3-q4",
        "name":     "Mistral 7B Instruct v0.3",
        "author":   "Mistral AI",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "url":      "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF"
                    "/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size_gb":  4.37,
        "ram_gb":   8,
        "context":  32768,
        "desc":     "Classic powerful model for text generation and analysis.",
        "tags":     ["powerful", "creative"],
    },
]

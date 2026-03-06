from pathlib import Path

APP_NAME    = "Local AI Runner"
APP_VERSION = "1.0.0"

MODELS_DIR = Path.home() / ".local_ai_runner" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CTX       = 4096
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMP      = 0.7
DEFAULT_TOP_P     = 0.95

MODEL_CATALOG = [
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

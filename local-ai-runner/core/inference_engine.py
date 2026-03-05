import os
import threading
import logging
from typing import Callable, Optional
from core.network_guard import NetworkGuard

logger = logging.getLogger(__name__)


class InferenceEngine:

    def __init__(self):
        self._llm             = None
        self._current_id: Optional[str] = None
        self._lock            = threading.Lock()
        self._running         = False
        self._stop_flag       = False

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def current_model_id(self) -> Optional[str]:
        return self._current_id

    # ── Load / Unload ─────────────────────────────────────────────────────────

    def load_model(
        self,
        model_path: str,
        model_id:   str,
        n_ctx:           int = 4096,
        n_threads:       Optional[int] = None,
        n_gpu_layers:    int = 0,
        on_complete: Callable[[], None]     = None,
        on_error:    Callable[[str], None]  = None,
    ) -> threading.Thread:

        def _worker():
            try:
                from llama_cpp import Llama
            except ImportError:
                msg = ("llama-cpp-python is not installed.\n"
                       "Run:  pip install llama-cpp-python")
                logger.error(msg)
                if on_error:
                    on_error(msg)
                return

            try:
                threads = n_threads or max(1, (os.cpu_count() or 2) - 1)
                logger.info(f"Loading {model_id} (threads={threads}, "
                            f"gpu_layers={n_gpu_layers}, ctx={n_ctx})")
                llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
                with self._lock:
                    self._llm        = llm
                    self._current_id = model_id
                logger.info(f"Model ready: {model_id}")
                if on_complete:
                    on_complete()
            except Exception as exc:
                logger.error(f"Load error: {exc}")
                if on_error:
                    on_error(str(exc))

        t = threading.Thread(target=_worker, daemon=True, name="model-load")
        t.start()
        return t

    def unload(self):
        with self._lock:
            self._llm        = None
            self._current_id = None
        logger.info("Model unloaded.")

    # ── Inference ────────────────────────────────────────────────────────────

    def generate(
        self,
        messages:     list[dict],
        on_token:     Callable[[str], None]   = None,
        on_complete:  Callable[[str], None]   = None,
        on_error:     Callable[[str], None]   = None,
        max_tokens:   int   = 2048,
        temperature:  float = 0.7,
        top_p:        float = 0.95,
    ) -> Optional[threading.Thread]:

        if not self._llm:
            if on_error:
                on_error("No model is loaded.")
            return None

        self._stop_flag = False

        def _worker():
            self._running = True
            collected: list[str] = []
            try:
                with NetworkGuard():            # blocks network for THIS thread
                    stream = self._llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=True,
                    )
                    for chunk in stream:
                        if self._stop_flag:
                            break
                        delta = (chunk["choices"][0]["delta"]
                                 .get("content", ""))
                        if delta:
                            collected.append(delta)
                            if on_token:
                                on_token(delta)

                if on_complete:
                    on_complete("".join(collected))

            except Exception as exc:
                logger.error(f"Inference error: {exc}")
                if on_error:
                    on_error(str(exc))
            finally:
                self._running = False

        t = threading.Thread(target=_worker, daemon=True, name="inference")
        t.start()
        return t

    def stop(self):
        self._stop_flag = True

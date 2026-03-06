import os
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
import config
from config import DEFAULT_CTX, DEFAULT_MAX_TOKENS, DEFAULT_TEMP, DEFAULT_TOP_P, MODELS_DIR


class SettingsPanel:

    def __init__(self, parent, engine):
        self.engine = engine
        self.frame  = ttk.Frame(parent)
        self._build()

    def _build(self):
        f = self.frame
        ttk.Label(f, text="Inference Settings",
                  font=("Segoe UI", 12, "bold"),
                  foreground="#cba6f7").pack(anchor="w", padx=16, pady=(12, 8))

        container = ttk.Frame(f)
        container.pack(padx=20, pady=4, anchor="w")

        cpu_count = max(1, (os.cpu_count() or 2) - 1)

        rows = [
            ("Context length (n_ctx)",   "n_ctx",        DEFAULT_CTX,        512,   131072, 512),
            ("Max new tokens",           "max_tokens",   DEFAULT_MAX_TOKENS,  64,   16384,  64),
            ("Top-P",                    "top_p",        DEFAULT_TOP_P,       0.01,  1.0,   0.01),
            ("CPU threads",              "n_threads",    cpu_count,           1,     64,    1),
            ("GPU layers (0 = CPU only)","n_gpu_layers", 0,                   0,    200,    1),
        ]

        self._vars: dict[str, tk.Variable] = {}

        for label, key, default, lo, hi, step in rows:
            row = ttk.Frame(container)
            row.pack(fill="x", pady=5)

            ttk.Label(row, text=label, width=30).pack(side="left")

            if isinstance(default, float):
                var = tk.DoubleVar(value=default)
            else:
                var = tk.IntVar(value=default)
            self._vars[key] = var

            val_lbl = ttk.Label(row, text=str(default), width=8)

            scl = ttk.Scale(row, from_=lo, to=hi, variable=var,
                            orient="horizontal", length=220)
            scl.pack(side="left", padx=8)
            val_lbl.pack(side="left")

            def _update_label(_, v=var, lbl=val_lbl, s=step):
                raw = v.get()
                if isinstance(raw, float) and s < 1:
                    lbl.config(text=f"{raw:.2f}")
                else:
                    lbl.config(text=str(int(raw)))

            var.trace_add("write", _update_label)

        # Privacy note
        note_frame = tk.Frame(f, bg="#1e3a1e", bd=0)
        note_frame.pack(fill="x", padx=16, pady=(20, 8))
        tk.Label(note_frame,
                 text="🔒  Privacy guarantee\n"
                      "Inference runs with NetworkGuard active.\n"
                      "All outgoing connections are blocked for the "
                      "inference thread.\n"
                      "Your prompts and responses never leave this machine.",
                 bg="#1e3a1e", fg="#a6e3a1",
                 font=("Segoe UI", 9), justify="left",
                 padx=12, pady=10).pack(anchor="w")

        # Default models directory chooser
        dir_row = ttk.Frame(f)
        dir_row.pack(fill="x", padx=16, pady=(6,12))
        ttk.Label(dir_row, text="Default models folder:", width=30).pack(side="left")
        self._models_dir_var = tk.StringVar(value=str(MODELS_DIR))
        dir_entry = ttk.Entry(dir_row, textvariable=self._models_dir_var, width=50)
        dir_entry.pack(side="left", padx=(0,8))
        def _choose_dir():
            sel = filedialog.askdirectory(title="Select default models folder")
            if sel:
                self._models_dir_var.set(sel)
                # update runtime config
                try:
                    p = Path(sel)
                    p.mkdir(parents=True, exist_ok=True)
                    config.MODELS_DIR = p
                except Exception:
                    pass

        ttk.Button(dir_row, text="Change…", command=_choose_dir).pack(side="left")

    def get_settings(self) -> dict:
        return {key: var.get() for key, var in self._vars.items()}

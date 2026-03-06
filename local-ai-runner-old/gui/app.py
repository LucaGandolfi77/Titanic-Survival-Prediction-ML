import tkinter as tk
from tkinter import ttk
from core.model_manager   import ModelManager
from core.inference_engine import InferenceEngine
from gui.chat_panel     import ChatPanel
from gui.model_panel    import ModelPanel
from gui.settings_panel import SettingsPanel
from config import APP_NAME, APP_VERSION


class Application:

    def __init__(self, root: tk.Tk):
        self.root    = root
        self.manager = ModelManager()
        self.engine  = InferenceEngine()

        self._configure_root()
        self._build_ui()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _configure_root(self):
        self.root.title(f"{APP_NAME}  v{APP_VERSION}")
        self.root.geometry("1100x720")
        self.root.minsize(800, 550)
        self.root.configure(bg="#1e1e2e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._apply_theme()

    def _apply_theme(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        bg, fg, acc = "#1e1e2e", "#cdd6f4", "#89b4fa"
        card = "#313244"
        style.configure(".",
                         background=bg, foreground=fg,
                         fieldbackground=card, troughcolor=card,
                         borderwidth=0, focuscolor=acc)
        style.configure("TNotebook",        background=bg, padding=0)
        style.configure("TNotebook.Tab",    background=card, foreground=fg,
                         padding=[14, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", acc)],
                  foreground=[("selected", "#1e1e2e")])
        style.configure("TFrame",           background=bg)
        style.configure("TLabel",           background=bg, foreground=fg)
        style.configure("TButton",
                         background=acc, foreground="#1e1e2e",
                         font=("Segoe UI", 9, "bold"), padding=[8, 4])
        style.map("TButton",
                  background=[("active", "#b4befe"), ("disabled", card)])
        style.configure("Ghost.TButton",
                         background=card, foreground=fg, padding=[8, 4])
        style.map("Ghost.TButton",
                  background=[("active", "#45475a")])
        style.configure("Danger.TButton",
                         background="#f38ba8", foreground="#1e1e2e",
                         font=("Segoe UI", 9, "bold"), padding=[8, 4])
        style.configure("TProgressbar",
                         troughcolor=card, background=acc, thickness=8)
        style.configure("TScrollbar",
                         background=card, troughcolor=bg, arrowcolor=fg)
        style.configure("TScale",
                         background=bg, troughcolor=card)
        style.configure("Status.TLabel",
                         background="#181825", foreground="#6c7086",
                         font=("Segoe UI", 9), padding=[8, 4])

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=0, pady=0)
        ttk.Label(top,
                  text=f"  🤖 {APP_NAME}",
                  font=("Segoe UI", 12, "bold"),
                  foreground="#cba6f7").pack(side="left", pady=8)

        self._net_label = ttk.Label(top,
                                    text="🔒 Offline-safe",
                                    foreground="#a6e3a1")
        self._net_label.pack(side="right", padx=16)

        # Notebook tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        self.chat_panel     = ChatPanel(self.notebook, self.engine,
                                        self.manager, self)
        self.model_panel    = ModelPanel(self.notebook, self.manager,
                                         self.engine, self)
        self.settings_panel = SettingsPanel(self.notebook, self.engine)

        self.notebook.add(self.chat_panel.frame,     text="💬  Chat")
        self.notebook.add(self.model_panel.frame,    text="📦  Models")
        self.notebook.add(self.settings_panel.frame, text="⚙️  Settings")

        # Status bar
        self._status_var = tk.StringVar(value="Ready — no model loaded.")
        ttk.Label(self.root, textvariable=self._status_var,
                  style="Status.TLabel").pack(fill="x", side="bottom")

    # ── Public helpers ────────────────────────────────────────────────────────

    def set_status(self, text: str):
        self._status_var.set(text)

    def refresh_chat_model_list(self):
        self.chat_panel.refresh_model_selector()

    def _on_close(self):
        self.engine.stop()
        self.root.destroy()

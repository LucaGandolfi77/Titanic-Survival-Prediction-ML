import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from config import MODEL_CATALOG


class ModelPanel:

    def __init__(self, parent, manager, engine, app):
        self.manager = manager
        self.engine  = engine
        self.app     = app
        self.frame   = ttk.Frame(parent)
        self._dl_threads: dict = {}
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        f = self.frame

        ttk.Label(f, text="Available Models",
                  font=("Segoe UI", 12, "bold"),
                  foreground="#cba6f7").pack(anchor="w", padx=14, pady=(10, 4))

        # Scrollable canvas for model cards
        outer = ttk.Frame(f)
        outer.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        canvas = tk.Canvas(outer, bg="#1e1e2e", highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical",
                                command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._cards_frame = ttk.Frame(canvas)
        self._cards_win = canvas.create_window(
            (0, 0), window=self._cards_frame, anchor="nw")

        self._cards_frame.bind("<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(
                self._cards_win, width=e.width))
        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        self._build_cards()

    def _build_cards(self):
        for widget in self._cards_frame.winfo_children():
            widget.destroy()

        self._progress_vars: dict[str, tk.DoubleVar] = {}
        self._progress_bars: dict[str, ttk.Progressbar] = {}
        self._status_labels: dict[str, ttk.Label] = {}
        self._dl_buttons:    dict[str, ttk.Button] = {}

        for m in MODEL_CATALOG:
            self._build_card(m)

    def _build_card(self, m: dict):
        mid     = m["id"]
        # consider downloaded if manager reports a local path (indexed or default)
        is_dl   = bool(self.manager.get_local_path(mid))
        card_bg = "#313244"

        card = tk.Frame(self._cards_frame, bg=card_bg,
                        highlightthickness=1,
                        highlightbackground="#45475a")
        card.pack(fill="x", padx=6, pady=5, ipady=6, ipadx=8)

        # ── Row 1: name + author + tags ──────────────────────────────────────
        row1 = tk.Frame(card, bg=card_bg)
        row1.pack(fill="x")

        tk.Label(row1, text=m["name"],
                 bg=card_bg, fg="#cdd6f4",
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Label(row1, text=f"  by {m['author']}",
                 bg=card_bg, fg="#6c7086",
                 font=("Segoe UI", 9)).pack(side="left")

        tag_str = "  " + "  ".join(f"[{t}]" for t in m.get("tags", []))
        tk.Label(row1, text=tag_str, bg=card_bg, fg="#89b4fa",
                 font=("Segoe UI", 8)).pack(side="left")

        # ── Row 2: description ────────────────────────────────────────────────
        tk.Label(card, text=m["desc"], bg=card_bg, fg="#a6adc8",
                 font=("Segoe UI", 9), anchor="w",
                 wraplength=700, justify="left").pack(fill="x", pady=(2, 4))

        # ── Row 3: specs + buttons ────────────────────────────────────────────
        row3 = tk.Frame(card, bg=card_bg)
        row3.pack(fill="x")

        specs = (f"💾 {m['size_gb']:.1f} GB   "
                 f"🧠 {m['ram_gb']} GB RAM   "
                 f"📏 {m['context']:,} ctx")
        tk.Label(row3, text=specs, bg=card_bg, fg="#585b70",
                 font=("Segoe UI", 8)).pack(side="left")

        # Buttons
        btn_frame = tk.Frame(card, bg=card_bg)
        btn_frame.pack(anchor="e")

        if is_dl:
            # Load button
            ttk.Button(btn_frame, text="⚡ Load",
                       command=lambda _mid=mid: self._load(
                           _mid)).pack(side="left", padx=2)
            # Delete button
            ttk.Button(btn_frame, text="🗑 Delete",
                       style="Danger.TButton",
                       command=lambda _mid=mid: self._delete(
                           _mid)).pack(side="left", padx=2)
            status_text = "✅ Downloaded"
            status_col  = "#a6e3a1"
        else:
            dl_btn = ttk.Button(btn_frame, text="⬇ Download",
                                command=lambda _mid=mid: self._download(
                                    _mid))
            dl_btn.pack(side="left", padx=2)
            self._dl_buttons[mid] = dl_btn

            # Allow the user to register a local GGUF file for this model
            def _choose_local():
                fp = filedialog.askopenfilename(
                    title=f"Select local GGUF for {m['name']}",
                    filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")])
                if not fp:
                    return
                ok = self.manager.register_local_path(mid, fp)
                if ok:
                    self._build_cards()
                    self.app.refresh_chat_model_list()
                    self.app.set_status(f"📁 Registered local file for {m['name']}")
                else:
                    messagebox.showerror("Error", "Failed to register local file.")

            ttk.Button(btn_frame, text="📁 Use local",
                       command=_choose_local).pack(side="left", padx=2)
            status_text = f"{m['size_gb']:.1f} GB"
            status_col  = "#6c7086"

        sl = ttk.Label(btn_frame, text=status_text, foreground=status_col)
        sl.pack(side="left", padx=6)
        self._status_labels[mid] = sl

        # Progress bar (hidden until download starts)
        pv = tk.DoubleVar(value=0)
        pb = ttk.Progressbar(card, variable=pv, maximum=100, length=300)
        self._progress_vars[mid] = pv
        self._progress_bars[mid] = pb

    # ── Actions ───────────────────────────────────────────────────────────────

    def _download(self, mid: str):
        btn = self._dl_buttons.get(mid)
        if btn:
            btn.config(state="disabled", text="⬇ Downloading…")

        pb = self._progress_bars[mid]
        pb.pack(fill="x", padx=6, pady=(2, 0))

        def on_progress(done: int, total: int):
            if total > 0:
                pct = done / total * 100
                self.frame.after(0, lambda p=pct: (
                    self._progress_vars[mid].set(p),
                    self._status_labels[mid].config(
                        text=f"{done/1_048_576:.0f}/{total/1_048_576:.0f} MB"),
                ))

        def on_complete(_path):
            self.frame.after(0, lambda: (
                self._build_cards(),
                self.app.refresh_chat_model_list(),
                self.app.set_status(
                    f"✅ Downloaded: {self.manager.get_by_id(mid)['name']}"),
            ))

        def on_error(msg):
            self.frame.after(0, lambda: (
                self._status_labels.get(mid, ttk.Label()).config(
                    text="❌ Error", foreground="#f38ba8"),
                self._progress_bars.get(mid, ttk.Progressbar()).pack_forget(),
                messagebox.showerror("Download Error", msg),
            ))

        # Ask user where to download (default: configured models dir)
        dest = filedialog.askdirectory(
            title="Choose download folder (Cancel to use default)")
        dest_arg = dest if dest else None

        self.manager.download(mid,
                              dest_dir=dest_arg,
                              on_progress=on_progress,
                              on_complete=on_complete,
                              on_error=on_error)

    def _load(self, mid: str):
        m    = self.manager.get_by_id(mid)
        path = self.manager.get_local_path(mid)
        if not path:
            messagebox.showerror("Error", "Model file not found.")
            return

        settings = self.app.settings_panel.get_settings()
        ctx      = min(settings["n_ctx"], m.get("context", 4096))

        self.app.set_status(f"Loading {m['name']}…")

        def on_done():
            self.frame.after(0, lambda: (
                self.app.set_status(f"✅ {m['name']} ready"),
                self.app.refresh_chat_model_list(),
            ))

        def on_err(msg):
            self.frame.after(0, lambda: messagebox.showerror(
                "Load Error", msg))

        self.engine.load_model(
            str(path), mid,
            n_ctx=ctx,
            n_threads=settings["n_threads"],
            n_gpu_layers=settings["n_gpu_layers"],
            on_complete=on_done,
            on_error=on_err,
        )

    def _delete(self, mid: str):
        m = self.manager.get_by_id(mid)
        if mid == self.engine.current_model_id:
            if not messagebox.askyesno(
                    "Delete", f"'{m['name']}' is currently loaded.\n"
                              "Unload and delete?"):
                return
            self.engine.unload()

        if self.manager.delete(mid):
            self._build_cards()
            self.app.refresh_chat_model_list()
            self.app.set_status(f"Deleted: {m['name']}")

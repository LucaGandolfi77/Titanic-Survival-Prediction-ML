import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from config import DEFAULT_TEMP, DEFAULT_MAX_TOKENS


class ChatPanel:

    def __init__(self, parent, engine, manager, app):
        self.engine  = engine
        self.manager = manager
        self.app     = app
        self.history: list[dict] = []   # [{role, content}, ...]
        self.frame   = ttk.Frame(parent)
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        f = self.frame

        # ── Top bar: model selector + controls ──────────────────────────────
        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=(8, 4))

        ttk.Label(top, text="Model:").pack(side="left")
        self._model_var = tk.StringVar(value="— no model loaded —")
        self._model_combo = ttk.Combobox(top, textvariable=self._model_var,
                                          state="readonly", width=34)
        self._model_combo.pack(side="left", padx=(4, 12))
        self._model_combo.bind("<<ComboboxSelected>>", self._on_model_select)

        ttk.Label(top, text="Temp:").pack(side="left")
        self._temp_var = tk.DoubleVar(value=DEFAULT_TEMP)
        ttk.Scale(top, from_=0.0, to=2.0, variable=self._temp_var,
                  orient="horizontal", length=90).pack(side="left", padx=2)
        self._temp_label = ttk.Label(top, text=f"{DEFAULT_TEMP:.1f}", width=3)
        self._temp_label.pack(side="left", padx=(0, 10))
        self._temp_var.trace_add("write",
            lambda *_: self._temp_label.config(
                text=f"{self._temp_var.get():.1f}"))

        self._clear_btn = ttk.Button(top, text="🗑 Clear",
                                      style="Ghost.TButton",
                                      command=self._clear_conversation)
        self._clear_btn.pack(side="right")

        self._status_lbl = ttk.Label(top, text="", foreground="#f38ba8")
        self._status_lbl.pack(side="right", padx=8)

        # ── System prompt (collapsible) ──────────────────────────────────────
        sys_frame = ttk.Frame(f)
        sys_frame.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(sys_frame, text="System prompt:",
                  font=("Segoe UI", 9)).pack(side="left")
        self._sys_text = tk.Text(sys_frame, height=2, wrap="word",
                                  bg="#313244", fg="#cdd6f4",
                                  insertbackground="#cdd6f4",
                                  relief="flat", font=("Segoe UI", 9),
                                  padx=6, pady=4)
        self._sys_text.insert("1.0",
            "You are a helpful, accurate assistant running fully offline.")
        self._sys_text.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._add_text_context_menu(self._sys_text)

        # ── Chat history ──────────────────────────────────────────────────────
        hist_frame = ttk.Frame(f)
        hist_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self._chat_text = tk.Text(
            hist_frame, wrap="word", state="disabled",
            bg="#1e1e2e", fg="#cdd6f4",
            font=("Segoe UI", 10), padx=10, pady=8,
            relief="flat", cursor="arrow",
        )
        self._chat_text.tag_config("user",
            foreground="#89dceb", font=("Segoe UI", 10, "bold"))
        self._chat_text.tag_config("assistant",
            foreground="#cdd6f4")
        self._chat_text.tag_config("system_msg",
            foreground="#6c7086", font=("Segoe UI", 9, "italic"))
        self._chat_text.tag_config("error",
            foreground="#f38ba8")

        scroll = ttk.Scrollbar(hist_frame, command=self._chat_text.yview)
        self._chat_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._chat_text.pack(side="left", fill="both", expand=True)
        self._add_text_context_menu(self._chat_text, readonly=True)

        # ── Input row ──────────────────────────────────────────────────────────
        inp_frame = ttk.Frame(f)
        inp_frame.pack(fill="x", padx=10, pady=(4, 10))

        self._input = tk.Text(
            inp_frame, height=3, wrap="word",
            bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief="flat", font=("Segoe UI", 10), padx=6, pady=6,
        )
        self._input.pack(side="left", fill="x", expand=True)
        self._add_text_context_menu(self._input)
        self._input.bind("<Control-Return>", lambda _: self._send())
        self._input.bind("<Return>", self._on_enter)
        self._input.focus_set()

        btn_col = ttk.Frame(inp_frame)
        btn_col.pack(side="right", padx=(6, 0))

        self._send_btn = ttk.Button(btn_col, text="Send  ↩",
                                     command=self._send, width=9)
        self._send_btn.pack(pady=(0, 4))

        self._stop_btn = ttk.Button(btn_col, text="⏹ Stop",
                                     style="Danger.TButton",
                                     command=self._stop, width=9,
                                     state="disabled")
        self._stop_btn.pack()

        self.refresh_model_selector()

    # ── Context menu (copy / cut / paste / select all) ────────────────────────

    def _add_text_context_menu(self, widget: tk.Text, readonly=False):
        menu = tk.Menu(widget, tearoff=0, bg="#313244", fg="#cdd6f4",
                       activebackground="#89b4fa",
                       activeforeground="#1e1e2e")

        menu.add_command(label="Copy",
            command=lambda: widget.event_generate("<<Copy>>"),
            accelerator="Ctrl+C")
        if not readonly:
            menu.add_command(label="Cut",
                command=lambda: widget.event_generate("<<Cut>>"),
                accelerator="Ctrl+X")
            menu.add_command(label="Paste",
                command=lambda: widget.event_generate("<<Paste>>"),
                accelerator="Ctrl+V")
        menu.add_separator()
        menu.add_command(label="Select All",
            command=lambda: (widget.tag_add("sel", "1.0", "end"),
                             widget.mark_set("insert", "end")),
            accelerator="Ctrl+A")

        def _show(event):
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        widget.bind("<Button-3>", _show)
        widget.bind("<Control-a>",
            lambda _: (widget.tag_add("sel", "1.0", "end"), "break"))

    # ── Model selector ────────────────────────────────────────────────────────

    def refresh_model_selector(self):
        downloaded = self.manager.list_downloaded()
        ids   = [m["id"]   for m in downloaded]
        names = [m["name"] for m in downloaded]
        self._model_ids = ids

        if names:
            self._model_combo["values"] = names
            current = self.engine.current_model_id
            if current in ids:
                self._model_var.set(names[ids.index(current)])
            else:
                self._model_var.set(names[0])
        else:
            self._model_combo["values"] = []
            self._model_var.set("— download a model first —")
            self._model_ids = []

    def _on_model_select(self, _event=None):
        sel = self._model_combo.current()
        if sel < 0 or sel >= len(self._model_ids):
            return
        model_id = self._model_ids[sel]
        if model_id == self.engine.current_model_id:
            return
        self._load_model(model_id)

    def _load_model(self, model_id: str):
        path = self.manager.get_local_path(model_id)
        if not path:
            messagebox.showerror("Error", f"Model file not found for: {model_id}")
            return

        m        = self.manager.get_by_id(model_id)
        settings = self.app.settings_panel.get_settings()
        ctx      = min(settings["n_ctx"], m.get("context", 4096))

        self._set_status(f"Loading {m['name']}…")
        self._send_btn.config(state="disabled")
        self.app.set_status(f"Loading {m['name']}…")

        def on_done():
            self.frame.after(0, lambda: (
                self._set_status(""),
                self._send_btn.config(state="normal"),
                self.app.set_status(f"✅  {m['name']} — ready"),
                self._append_system(f"Model loaded: {m['name']}"),
            ))

        def on_err(msg):
            self.frame.after(0, lambda: (
                self._set_status(f"Error: {msg}"),
                self._send_btn.config(state="normal"),
                messagebox.showerror("Load Error", msg),
            ))

        self.engine.load_model(
            str(path), model_id,
            n_ctx=ctx,
            n_threads=settings["n_threads"],
            n_gpu_layers=settings["n_gpu_layers"],
            on_complete=on_done,
            on_error=on_err,
        )

    # ── Sending / receiving ───────────────────────────────────────────────────

    def _on_enter(self, event):
        if event.state & 0x4:      # Ctrl+Enter = send
            self._send()
            return "break"
        # plain Enter = newline (default behaviour)

    def _send(self):
        if not self.engine.is_loaded:
            messagebox.showwarning("No Model",
                "Please load a model first (Models tab).")
            return

        raw = self._input.get("1.0", "end-1c").strip()
        if not raw:
            return

        self._input.delete("1.0", "end")
        self.history.append({"role": "user", "content": raw})
        self._append_user(raw)

        sys_prompt = self._sys_text.get("1.0", "end-1c").strip()
        messages   = ([{"role": "system", "content": sys_prompt}]
                      if sys_prompt else []) + self.history

        settings = self.app.settings_panel.get_settings()
        self._send_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._append_assistant_start()

        def on_token(token: str):
            self.frame.after(0, lambda t=token: self._append_token(t))

        def on_done(full: str):
            self.history.append({"role": "assistant", "content": full})
            self.frame.after(0, self._on_inference_done)

        def on_err(msg: str):
            self.frame.after(0, lambda: (
                self._append_error(f"\n[Error: {msg}]"),
                self._on_inference_done(),
            ))

        self.engine.generate(
            messages,
            on_token=on_token,
            on_complete=on_done,
            on_error=on_err,
            max_tokens=settings["max_tokens"],
            temperature=self._temp_var.get(),
            top_p=settings["top_p"],
        )

    def _stop(self):
        self.engine.stop()
        self._on_inference_done()

    def _on_inference_done(self):
        self._send_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # ── Chat display helpers ──────────────────────────────────────────────────

    def _write(self, text: str, tag: str = ""):
        self._chat_text.config(state="normal")
        self._chat_text.insert("end", text, tag)
        self._chat_text.config(state="disabled")
        self._chat_text.see("end")

    def _append_user(self, text: str):
        self._write(f"\n👤 You\n{text}\n\n", "user")

    def _append_assistant_start(self):
        self._write("🤖 Assistant\n", "assistant")

    def _append_token(self, token: str):
        self._write(token, "assistant")
        # add trailing newlines after last token is shown via on_done

    def _append_error(self, text: str):
        self._write(text, "error")

    def _append_system(self, text: str):
        self._write(f"\n── {text} ──\n", "system_msg")

    def _clear_conversation(self):
        if not self.history:
            return
        if messagebox.askyesno("Clear", "Clear conversation history?"):
            self.history.clear()
            self._chat_text.config(state="normal")
            self._chat_text.delete("1.0", "end")
            self._chat_text.config(state="disabled")

    def _set_status(self, text: str):
        self._status_lbl.config(text=text)

"""
PDF Region Cropper
------------------
Select a rectangular region on any PDF page preview,
then export that region as a compressed image (JPEG/PNG)
for every page in the PDF.

Requirements:
    pip install pymupdf Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# PyMuPDF exposes itself as ``fitz`` but some Python environments
# also have a completely unrelated ``frontend`` package that
# blindly pulls in Starlette and tries to mount ``static/``.  On a
# fresh conda install that directory may be missing, which causes
# import to blow up with
#
#     RuntimeError: Directory 'static/' does not exist
#
# The workaround below is simply to ensure the path exists before
# importing fitz and to translate any import error into a clearer
# message for the user.
import os

# ensure a benign ``static/`` exists in CWD (Starlette uses cwd
# relative paths when importing the problematic ``frontend`` package).
try:
    os.makedirs("static", exist_ok=True)
except Exception:
    pass

try:
    import fitz  # type: ignore[import-not-found] # PyMuPDF
    # the PyMuPDF distribution defines Document/Rect/etc.; the bogus
    # ``fitz`` package from PyPI does not.  Help the user by detecting the
    # wrong library early.
    if not hasattr(fitz, "Document"):
        raise ImportError("installed `fitz` is not PyMuPDF")
except Exception as exc:  # could be RuntimeError or ImportError
    msg = (
        "Unable to import PyMuPDF (fitz).\n"
        "It looks like your environment has a conflicting package "
        "named ``fitz`` (https://pypi.org/project/fitz) which is *not* "
        "PyMuPDF.\n"
        "\nTo fix this, uninstall the incorrect package and install the right one:\n"
        "    pip uninstall fitz frontend\n"
        "    pip install --upgrade pymupdf\n"
        "\nIf you still see a message about ``'static/' does not exist``, "
        "make sure an empty ``static/`` directory exists in your working "
        "directory (or site-packages/frontend) or completely remove the "
        "``frontend`` package."
    )
    raise ImportError(msg) from exc

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    msg = (
        "Unable to import PIL (Pillow).\n"
        "Please install it with:\n"
        "    pip install --upgrade Pillow\n"
    )
    raise ImportError(msg) from exc

import io
import threading


# ─────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────

def pdf_page_to_pil(page: fitz.Page, zoom: float = 1.5) -> Image.Image:
    """Render a PDF page to a PIL Image at the given zoom level."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def crop_and_compress(
    page: fitz.Page,
    pdf_rect: fitz.Rect,
    fmt: str,
    quality: int,
    dpi: int,
) -> bytes:
    """
    Render a clipped region of a PDF page and return compressed bytes.

    Args:
        page:     PyMuPDF page object.
        pdf_rect: Crop rectangle in PDF coordinate space.
        fmt:      Output format: 'JPEG' or 'PNG'.
        quality:  JPEG quality 1–100 (ignored for PNG).
        dpi:      Render resolution.

    Returns:
        Compressed image bytes.
    """
    zoom = dpi / 72.0          # 72 points-per-inch is the PDF default
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=pdf_rect, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────

class PDFCropperApp(tk.Tk):
    # Maximum canvas dimension (preview fits inside this box)
    MAX_PREVIEW = 700

    def __init__(self):
        super().__init__()
        self.title("PDF Region Cropper")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        # ── State ──────────────────────────────────────
        self.doc: fitz.Document | None = None
        self.current_page_idx: int = 0
        self.preview_img: ImageTk.PhotoImage | None = None
        self.preview_pil: Image.Image | None = None
        self.zoom_factor: float = 1.0  # canvas px / pdf pt

        # Crop rectangle in canvas space (set while dragging)
        self._drag_start: tuple[int, int] | None = None
        self._rect_id: int | None = None

        # Crop rectangle stored in PDF coordinate space
        self.crop_rect_pdf: fitz.Rect | None = None

        self._build_ui()

    # ─────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────

    def _build_ui(self):
        """Build all UI widgets."""
        # ── Colour palette ─────────────────────────────
        BG     = "#1e1e2e"
        PANEL  = "#2a2a3e"
        ACCENT = "#7c3aed"
        TEXT   = "#e2e8f0"
        ENTRY  = "#313150"

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",      background=PANEL)
        style.configure("TLabel",      background=PANEL,  foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TButton",     background=ACCENT, foreground="white",
                        font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton",           background=[("active", "#6d28d9")])
        style.configure("TScale",      background=PANEL, troughcolor=ENTRY)
        style.configure("TCombobox",   fieldbackground=ENTRY, background=PANEL,
                        foreground=TEXT, selectbackground=ACCENT)
        style.configure("Horizontal.TProgressbar",
                        troughcolor=ENTRY, background=ACCENT, thickness=12)

        # ── Root grid ──────────────────────────────────
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Left sidebar
        sidebar = ttk.Frame(self, width=260)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 4), pady=10)
        sidebar.columnconfigure(0, weight=1)
        sidebar.grid_propagate(False)

        # Right canvas area
        canvas_frame = ttk.Frame(self)
        canvas_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 10), pady=10)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # ── Sidebar widgets ────────────────────────────
        row = 0

        # Title
        tk.Label(sidebar, text="PDF Region Cropper",
                 bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).grid(
            row=row, column=0, pady=(10, 16), padx=10)
        row += 1

        # Open PDF button
        ttk.Button(sidebar, text="📂  Open PDF",
                   command=self._open_pdf).grid(
            row=row, column=0, sticky="ew", padx=10)
        row += 1

        # PDF info
        self.lbl_pdf = tk.Label(sidebar, text="No file loaded",
                                bg=PANEL, fg="#94a3b8",
                                font=("Segoe UI", 9), wraplength=220, justify="left")
        self.lbl_pdf.grid(row=row, column=0, sticky="w", padx=10, pady=(4, 12))
        row += 1

        ttk.Separator(sidebar, orient="horizontal").grid(
            row=row, column=0, sticky="ew", padx=10, pady=4)
        row += 1

        # Page navigation
        tk.Label(sidebar, text="Preview Page",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=10)
        row += 1

        nav_frame = ttk.Frame(sidebar)
        nav_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(2, 10))
        nav_frame.columnconfigure(1, weight=1)
        row += 1

        ttk.Button(nav_frame, text="◀", width=3,
                   command=self._prev_page).grid(row=0, column=0)
        self.lbl_page = tk.Label(nav_frame, text="—",
                                 bg=PANEL, fg=TEXT, font=("Segoe UI", 10))
        self.lbl_page.grid(row=0, column=1)
        ttk.Button(nav_frame, text="▶", width=3,
                   command=self._next_page).grid(row=0, column=2)

        ttk.Separator(sidebar, orient="horizontal").grid(
            row=row, column=0, sticky="ew", padx=10, pady=4)
        row += 1

        # Format selector
        tk.Label(sidebar, text="Output Format",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=10)
        row += 1

        self.var_fmt = tk.StringVar(value="JPEG")
        fmt_frame = ttk.Frame(sidebar)
        fmt_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(2, 10))
        row += 1
        for fmt in ("JPEG", "PNG"):
            tk.Radiobutton(fmt_frame, text=fmt, variable=self.var_fmt,
                           value=fmt, bg=PANEL, fg=TEXT,
                           selectcolor=ACCENT, activebackground=PANEL,
                           font=("Segoe UI", 10),
                           command=self._on_format_change).pack(side="left", padx=8)

        # JPEG quality slider
        self.frm_quality = ttk.Frame(sidebar)
        self.frm_quality.grid(row=row, column=0, sticky="ew", padx=10)
        row += 1

        tk.Label(self.frm_quality, text="JPEG Quality",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w")
        self.var_quality = tk.IntVar(value=85)
        self.lbl_quality_val = tk.Label(self.frm_quality, text="85",
                                        bg=PANEL, fg=ACCENT,
                                        font=("Segoe UI", 10, "bold"), width=3)
        self.lbl_quality_val.grid(row=0, column=1, sticky="e")
        self.frm_quality.columnconfigure(0, weight=1)

        sld = ttk.Scale(self.frm_quality, from_=10, to=100,
                        orient="horizontal", variable=self.var_quality,
                        command=self._on_quality_change)
        sld.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        # Render DPI
        ttk.Separator(sidebar, orient="horizontal").grid(
            row=row, column=0, sticky="ew", padx=10, pady=4)
        row += 1

        tk.Label(sidebar, text="Render DPI",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=10)
        row += 1

        self.var_dpi = tk.StringVar(value="150")
        dpi_combo = ttk.Combobox(sidebar, textvariable=self.var_dpi,
                                 values=["72", "96", "150", "200", "300", "600"],
                                 state="readonly", width=8)
        dpi_combo.grid(row=row, column=0, sticky="w", padx=10, pady=(2, 10))
        row += 1

        ttk.Separator(sidebar, orient="horizontal").grid(
            row=row, column=0, sticky="ew", padx=10, pady=4)
        row += 1

        # Crop info
        tk.Label(sidebar, text="Selected Region",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=10)
        row += 1

        self.lbl_crop = tk.Label(sidebar, text="Draw a rectangle on the preview",
                                 bg=PANEL, fg="#94a3b8",
                                 font=("Segoe UI", 9), wraplength=220, justify="left")
        self.lbl_crop.grid(row=row, column=0, sticky="w", padx=10, pady=(2, 4))
        row += 1

        ttk.Button(sidebar, text="✖  Clear Selection",
                   command=self._clear_selection).grid(
            row=row, column=0, sticky="ew", padx=10, pady=(2, 10))
        row += 1

        ttk.Separator(sidebar, orient="horizontal").grid(
            row=row, column=0, sticky="ew", padx=10, pady=4)
        row += 1

        # Output folder
        tk.Label(sidebar, text="Output Folder",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=10)
        row += 1

        out_frame = ttk.Frame(sidebar)
        out_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(2, 4))
        out_frame.columnconfigure(0, weight=1)
        row += 1

        self.var_outdir = tk.StringVar(value="")
        tk.Entry(out_frame, textvariable=self.var_outdir,
                 bg=ENTRY, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 9)).grid(
            row=0, column=0, sticky="ew", ipady=4)
        ttk.Button(out_frame, text="…", width=3,
                   command=self._choose_outdir).grid(row=0, column=1, padx=(4, 0))

        # Progress bar
        self.progress = ttk.Progressbar(sidebar, orient="horizontal",
                                        mode="determinate",
                                        style="Horizontal.TProgressbar")
        self.progress.grid(row=row, column=0, sticky="ew", padx=10, pady=(4, 2))
        row += 1

        self.lbl_progress = tk.Label(sidebar, text="",
                                     bg=PANEL, fg="#94a3b8",
                                     font=("Segoe UI", 9))
        self.lbl_progress.grid(row=row, column=0, sticky="w", padx=10)
        row += 1

        # Export button
        self.btn_export = ttk.Button(sidebar, text="🚀  Export All Pages",
                                     command=self._start_export)
        self.btn_export.grid(row=row, column=0, sticky="ew",
                             padx=10, pady=(8, 10))
        row += 1

        # ── Canvas ─────────────────────────────────────
        self.canvas = tk.Canvas(canvas_frame, bg="#12121f",
                                cursor="crosshair",
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Scrollbars
        sb_y = ttk.Scrollbar(canvas_frame, orient="vertical",
                              command=self.canvas.yview)
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x = ttk.Scrollbar(canvas_frame, orient="horizontal",
                              command=self.canvas.xview)
        sb_x.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)

        # Instructions overlay
        self.canvas.create_text(
            350, 250,
            text="Open a PDF to begin.\n\nDrag on the preview to select\nthe region to crop.",
            fill="#4a4a7a", font=("Segoe UI", 14), justify="center",
            tags="hint"
        )

        # Mouse bindings for drawing crop rectangle
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",        self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_mouse_release)
        self.canvas.bind("<MouseWheel>",       self._on_mousewheel)

    # ─────────────────────────────────────────────────────
    # PDF loading
    # ─────────────────────────────────────────────────────

    def _open_pdf(self):
        path = filedialog.askopenfilename(
            title="Open PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            self.doc = fitz.open(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot open PDF:\n{exc}")
            return

        self.current_page_idx = 0
        self.crop_rect_pdf = None
        fname = os.path.basename(path)
        self.lbl_pdf.config(
            text=f"{fname}\n{len(self.doc)} pages",
            fg="#e2e8f0"
        )
        # Set default output folder next to the PDF
        self.var_outdir.set(os.path.dirname(path))
        self._render_page()

    def _render_page(self):
        if self.doc is None:
            return

        page = self.doc.load_page(self.current_page_idx)
        total = len(self.doc)
        self.lbl_page.config(
            text=f"Page {self.current_page_idx + 1} / {total}"
        )

        # Choose zoom so the page fits MAX_PREVIEW
        pw, ph = page.rect.width, page.rect.height
        zoom_x = self.MAX_PREVIEW / pw
        zoom_y = self.MAX_PREVIEW / ph
        zoom = min(zoom_x, zoom_y, 2.0)   # never enlarge beyond 2×
        self.zoom_factor = zoom

        pil = pdf_page_to_pil(page, zoom=zoom)
        self.preview_pil = pil
        self.preview_img = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, pil.width, pil.height))
        self.canvas.create_image(0, 0, anchor="nw",
                                  image=self.preview_img, tags="page")
        self._rect_id = None

        # Re-draw stored crop rect if present
        if self.crop_rect_pdf is not None:
            self._draw_stored_rect()

    def _draw_stored_rect(self):
        """Draw the stored PDF-space rect onto the current canvas."""
        r = self.crop_rect_pdf
        z = self.zoom_factor
        x0, y0, x1, y1 = r.x0 * z, r.y0 * z, r.x1 * z, r.y1 * z
        self._rect_id = self.canvas.create_rectangle(
            x0, y0, x1, y1,
            outline="#7c3aed", width=2, dash=(6, 3), tags="crop_rect"
        )

    # ─────────────────────────────────────────────────────
    # Page navigation
    # ─────────────────────────────────────────────────────

    def _prev_page(self):
        if self.doc and self.current_page_idx > 0:
            self.current_page_idx -= 1
            self._render_page()

    def _next_page(self):
        if self.doc and self.current_page_idx < len(self.doc) - 1:
            self.current_page_idx += 1
            self._render_page()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

    # ─────────────────────────────────────────────────────
    # Crop rectangle drawing
    # ─────────────────────────────────────────────────────

    def _on_mouse_press(self, event):
        if self.doc is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        self._drag_start = (cx, cy)
        # Remove previous rectangle
        self.canvas.delete("crop_rect")
        self._rect_id = None

    def _on_mouse_drag(self, event):
        if self._drag_start is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        x0, y0 = self._drag_start
        self.canvas.delete("crop_rect")
        self._rect_id = self.canvas.create_rectangle(
            x0, y0, cx, cy,
            outline="#7c3aed", width=2, dash=(6, 3), tags="crop_rect"
        )

    def _on_mouse_release(self, event):
        if self._drag_start is None or self.doc is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        x0, y0 = self._drag_start
        self._drag_start = None

        # Normalise coordinates
        rx0, rx1 = sorted([x0, cx])
        ry0, ry1 = sorted([y0, cy])

        # Reject too-small rectangles (< 5px)
        if (rx1 - rx0) < 5 or (ry1 - ry0) < 5:
            self.lbl_crop.config(text="Selection too small — try again.")
            return

        # Convert canvas px → PDF points
        z = self.zoom_factor
        pdf_x0 = rx0 / z
        pdf_y0 = ry0 / z
        pdf_x1 = rx1 / z
        pdf_y1 = ry1 / z

        # Clamp to page bounds
        page = self.doc.load_page(self.current_page_idx)
        pr = page.rect
        pdf_x0 = max(0.0, min(pdf_x0, pr.width))
        pdf_y0 = max(0.0, min(pdf_y0, pr.height))
        pdf_x1 = max(0.0, min(pdf_x1, pr.width))
        pdf_y1 = max(0.0, min(pdf_y1, pr.height))

        self.crop_rect_pdf = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)

        w_pt = pdf_x1 - pdf_x0
        h_pt = pdf_y1 - pdf_y0
        self.lbl_crop.config(
            text=f"x0={pdf_x0:.0f}  y0={pdf_y0:.0f}\n"
                 f"x1={pdf_x1:.0f}  y1={pdf_y1:.0f}\n"
                 f"Size: {w_pt:.0f} × {h_pt:.0f} pt"
        )

    def _clear_selection(self):
        self.crop_rect_pdf = None
        self.canvas.delete("crop_rect")
        self.lbl_crop.config(text="Draw a rectangle on the preview")

    # ─────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────

    def _choose_outdir(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.var_outdir.set(d)

    def _on_format_change(self):
        fmt = self.var_fmt.get()
        state = "normal" if fmt == "JPEG" else "disabled"
        for child in self.frm_quality.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _on_quality_change(self, value):
        self.lbl_quality_val.config(text=str(int(float(value))))

    def _start_export(self):
        if self.doc is None:
            messagebox.showwarning("Warning", "Please open a PDF first.")
            return
        if self.crop_rect_pdf is None:
            messagebox.showwarning("Warning",
                                   "Please draw a crop region on the preview.")
            return
        outdir = self.var_outdir.get().strip()
        if not outdir:
            messagebox.showwarning("Warning", "Please select an output folder.")
            return
        os.makedirs(outdir, exist_ok=True)

        # Disable button while exporting
        self.btn_export.configure(state="disabled")
        t = threading.Thread(target=self._export_worker,
                             args=(outdir,), daemon=True)
        t.start()

    def _export_worker(self, outdir: str):
        doc = self.doc
        rect = self.crop_rect_pdf
        fmt = self.var_fmt.get()
        quality = int(self.var_quality.get())
        dpi = int(self.var_dpi.get())
        total = len(doc)
        ext = "jpg" if fmt == "JPEG" else "png"

        errors = []
        for i in range(total):
            try:
                page = doc.load_page(i)
                data = crop_and_compress(page, rect, fmt, quality, dpi)
                fname = os.path.join(outdir, f"page_{i + 1:04d}.{ext}")
                with open(fname, "wb") as f:
                    f.write(data)
            except Exception as exc:
                errors.append(f"Page {i+1}: {exc}")

            # Update progress bar in main thread
            progress_pct = int((i + 1) / total * 100)
            self.after(0, self._update_progress,
                       progress_pct, i + 1, total)

        self.after(0, self._export_done, outdir, total, errors)

    def _update_progress(self, pct: int, current: int, total: int):
        self.progress["value"] = pct
        self.lbl_progress.config(text=f"Processing {current}/{total}…")

    def _export_done(self, outdir: str, total: int, errors: list[str]):
        self.btn_export.configure(state="normal")
        self.progress["value"] = 100
        if errors:
            self.lbl_progress.config(text=f"Done with {len(errors)} error(s)")
            messagebox.showwarning(
                "Export completed with errors",
                f"Exported {total - len(errors)}/{total} pages.\n\n"
                + "\n".join(errors[:10])
            )
        else:
            self.lbl_progress.config(text=f"✅ Done — {total} images saved")
            messagebox.showinfo(
                "Export complete",
                f"✅  {total} images saved to:\n{outdir}"
            )


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = PDFCropperApp()
    app.mainloop()

"""Tkinter GUI for WebScraper Pro.

Provides a desktop application with:
* URL input with validation
* Max-pages spinner + "Download images" checkbox
* Output-folder picker
* Real-time progress bar
* Live colour-coded log area
* Start / Stop / Open Excel / Open Folder buttons
* Background-threaded scraping (non-blocking UI)

Launch directly::

    python gui/app.py

Or as a module::

    python -m gui.app
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

# ── Ensure project root is on sys.path ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.image_downloader import run_download
from core.scraper import ProductScraper
from exporters.csv_exporter import CsvExporter
from exporters.excel_exporter import ExcelExporter
from utils.logger import get_logger
from utils.validator import is_valid_url, normalise_url

log = get_logger(__name__)

# ── Constants ───────────────────────────────────────────────────
_WINDOW_TITLE = "\U0001F577\uFE0F  WebScraper Pro — Product Extractor"
_MIN_WIDTH = 640
_MIN_HEIGHT = 520
_PAD = 8

_TAG_INFO = "info"
_TAG_WARN = "warn"
_TAG_ERROR = "error"


class WebScraperApp:
    """Main application window.

    Attributes:
        root: The Tk root window.
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(_WINDOW_TITLE)
        self.root.minsize(_MIN_WIDTH, _MIN_HEIGHT)
        self.root.resizable(True, True)

        # ── State ───────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._output_folder = tk.StringVar(value=str(Path("./output").resolve()))
        self._url_var = tk.StringVar()
        self._max_pages_var = tk.IntVar(value=50)
        self._download_images_var = tk.BooleanVar(value=True)
        self._excel_path: str = ""

        self._build_ui()

    # ── UI Construction ─────────────────────────────────────────

    def _build_ui(self) -> None:
        """Lay out all widgets."""

        # ── URL frame ───────────────────────────────────────────
        frm_url = ttk.LabelFrame(self.root, text="Target", padding=_PAD)
        frm_url.pack(fill=tk.X, padx=_PAD, pady=(_PAD, 4))

        ttk.Label(frm_url, text="URL:").grid(row=0, column=0, sticky=tk.W)
        self._url_entry = ttk.Entry(frm_url, textvariable=self._url_var, width=60)
        self._url_entry.grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=(4, 0))

        ttk.Label(frm_url, text="Max pages:").grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        self._pages_spin = ttk.Spinbox(
            frm_url, from_=1, to=999, textvariable=self._max_pages_var, width=6
        )
        self._pages_spin.grid(row=1, column=1, sticky=tk.W, padx=(4, 0), pady=(4, 0))

        self._img_check = ttk.Checkbutton(
            frm_url, text="Download images", variable=self._download_images_var
        )
        self._img_check.grid(row=1, column=2, sticky=tk.W, padx=(12, 0), pady=(4, 0))

        frm_url.columnconfigure(1, weight=1)

        # ── Output frame ────────────────────────────────────────
        frm_out = ttk.LabelFrame(self.root, text="Output", padding=_PAD)
        frm_out.pack(fill=tk.X, padx=_PAD, pady=4)

        ttk.Label(frm_out, text="Folder:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frm_out, textvariable=self._output_folder, width=50).grid(
            row=0, column=1, sticky=tk.EW, padx=(4, 4)
        )
        ttk.Button(frm_out, text="Browse…", command=self._browse_folder).grid(
            row=0, column=2
        )
        frm_out.columnconfigure(1, weight=1)

        # ── Action buttons ──────────────────────────────────────
        frm_act = ttk.Frame(self.root, padding=_PAD)
        frm_act.pack(fill=tk.X, padx=_PAD)

        self._btn_start = ttk.Button(frm_act, text="▶  Start Scraping", command=self._start)
        self._btn_start.pack(side=tk.LEFT)

        self._btn_stop = ttk.Button(frm_act, text="■  Stop", command=self._stop, state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT, padx=(8, 0))

        self._btn_open_excel = ttk.Button(
            frm_act, text="📄  Open Excel", command=self._open_excel, state=tk.DISABLED
        )
        self._btn_open_excel.pack(side=tk.RIGHT)

        self._btn_open_folder = ttk.Button(
            frm_act, text="📂  Open Folder", command=self._open_folder
        )
        self._btn_open_folder.pack(side=tk.RIGHT, padx=(0, 8))

        # ── Progress ────────────────────────────────────────────
        frm_prog = ttk.LabelFrame(self.root, text="Progress", padding=_PAD)
        frm_prog.pack(fill=tk.X, padx=_PAD, pady=4)

        self._progress = ttk.Progressbar(frm_prog, mode="determinate", maximum=100)
        self._progress.pack(fill=tk.X)

        self._lbl_status = ttk.Label(frm_prog, text="Idle")
        self._lbl_status.pack(anchor=tk.W, pady=(4, 0))

        # ── Log area ────────────────────────────────────────────
        frm_log = ttk.LabelFrame(self.root, text="Log", padding=_PAD)
        frm_log.pack(fill=tk.BOTH, expand=True, padx=_PAD, pady=(4, _PAD))

        self._log_text = tk.Text(frm_log, height=12, state=tk.DISABLED, wrap=tk.WORD,
                                 font=("Consolas", 9))
        scroll = ttk.Scrollbar(frm_log, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Colour tags
        self._log_text.tag_configure(_TAG_INFO, foreground="#228B22")
        self._log_text.tag_configure(_TAG_WARN, foreground="#DAA520")
        self._log_text.tag_configure(_TAG_ERROR, foreground="#DC143C")

    # ── Actions ─────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self._output_folder.get())
        if folder:
            self._output_folder.set(folder)

    def _start(self) -> None:
        """Validate inputs and launch the scraping thread."""
        raw_url = normalise_url(self._url_var.get())
        if not is_valid_url(raw_url):
            self._url_entry.configure(style="Error.TEntry")
            messagebox.showerror("Invalid URL", "Please enter a valid HTTP(S) URL.")
            return

        self._url_var.set(raw_url)
        self._stop_event.clear()

        self._btn_start.configure(state=tk.DISABLED)
        self._btn_stop.configure(state=tk.NORMAL)
        self._btn_open_excel.configure(state=tk.DISABLED)
        self._progress["value"] = 0
        self._clear_log()
        self._log("Starting scrape: " + raw_url, _TAG_INFO)

        self._worker_thread = threading.Thread(target=self._run_scraper, daemon=True)
        self._worker_thread.start()

    def _stop(self) -> None:
        """Request graceful stop."""
        self._stop_event.set()
        self._log("Stop requested — finishing current page …", _TAG_WARN)

    def _open_excel(self) -> None:
        if self._excel_path:
            self._open_file(self._excel_path)

    def _open_folder(self) -> None:
        folder = self._output_folder.get()
        self._open_file(folder)

    # ── Scraper worker (background thread) ──────────────────────

    def _run_scraper(self) -> None:
        """Execute the full scraping pipeline in a background thread."""
        try:
            url = self._url_var.get()
            max_pages = self._max_pages_var.get()
            download_images = self._download_images_var.get()
            output_folder = Path(self._output_folder.get())
            output_folder.mkdir(parents=True, exist_ok=True)

            config_path = _PROJECT_ROOT / "config" / "config.yaml"

            scraper = ProductScraper(
                config_path=str(config_path),
                stop_event=self._stop_event,
            )
            # Override max_pages from GUI
            scraper._max_pages = max_pages

            def on_page(page: int, items: int) -> None:
                pct = min(page / max_pages * 100, 100)
                self.root.after(0, self._update_progress, pct,
                                f"Page {page} — {items} products found")
                self.root.after(0, self._log,
                                f"Page {page}: {items} products so far", _TAG_INFO)

            products = scraper.scrape_all_pages(url, callback=on_page)

            if not products:
                self.root.after(0, self._log, "No products found.", _TAG_WARN)
                self.root.after(0, self._done, 0)
                return

            # Download images
            if download_images:
                self.root.after(0, self._log, "Downloading images …", _TAG_INFO)
                images_folder = output_folder / "images"
                products = run_download(
                    products,
                    output_folder=str(images_folder),
                    max_concurrent=5,
                )

            # Export Excel
            excel_path = output_folder / "products.xlsx"
            exporter = ExcelExporter(excel_path)
            self._excel_path = exporter.export(
                products,
                source_url=url,
                pages_visited=scraper.pages_visited,
            )

            # Export CSV
            csv_path = output_folder / "products.csv"
            CsvExporter(csv_path).export(products)

            self.root.after(0, self._log,
                            f"Saved {len(products)} products → {self._excel_path}", _TAG_INFO)
            self.root.after(0, self._done, len(products))
            scraper.close()

        except Exception as exc:  # noqa: BLE001
            log.exception("Scraper error")
            self.root.after(0, self._log, f"ERROR: {exc}", _TAG_ERROR)
            self.root.after(0, self._done, -1)

    # ── UI helpers ──────────────────────────────────────────────

    def _update_progress(self, pct: float, text: str) -> None:
        self._progress["value"] = pct
        self._lbl_status.configure(text=text)

    def _done(self, count: int) -> None:
        self._btn_start.configure(state=tk.NORMAL)
        self._btn_stop.configure(state=tk.DISABLED)
        self._progress["value"] = 100 if count > 0 else 0

        if count > 0:
            self._btn_open_excel.configure(state=tk.NORMAL)
            self._lbl_status.configure(text=f"Done — {count} products exported")
            messagebox.showinfo(
                "Scraping complete!",
                f"{count} products saved to\n{self._excel_path}",
            )
        elif count == 0:
            self._lbl_status.configure(text="No products found")
        else:
            self._lbl_status.configure(text="Stopped with errors")

    def _log(self, message: str, tag: str = _TAG_INFO) -> None:
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, message + "\n", tag)
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.configure(state=tk.DISABLED)

    @staticmethod
    def _open_file(path: str) -> None:
        """Open a file or folder with the OS default handler."""
        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
            elif system == "Darwin":
                subprocess.Popen(["open", path])  # noqa: S603, S607
            else:
                subprocess.Popen(["xdg-open", path])  # noqa: S603, S607
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not open %s: %s", path, exc)

    # ── Entry point ─────────────────────────────────────────────

    def run(self) -> None:
        """Start the Tkinter main-loop."""
        self.root.mainloop()


def main() -> None:
    """Launch the WebScraper Pro GUI."""
    app = WebScraperApp()
    app.run()


if __name__ == "__main__":
    main()

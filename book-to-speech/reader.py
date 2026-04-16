"""
reader.py — Unified book file reader.

Supports:
  • .txt / .md  — plain text
  • .pdf         — via pypdf
  • .epub        — via ebooklib + BeautifulSoup
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup


def read_file(path: str | Path) -> str:
    """
    Read a book file and return its full text as a string.

    Supported formats: .txt, .md, .pdf, .epub
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return _read_text(path)
    elif suffix == ".pdf":
        return _read_pdf(path)
    elif suffix == ".epub":
        return _read_epub(path)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            "Supported: .txt, .md, .pdf, .epub"
        )


def _read_text(path: Path) -> str:
    """Read plain-text / markdown files."""
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF files. "
            "Install it: pip install pypdf"
        )

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n\n".join(pages)


def _read_epub(path: Path) -> str:
    """Extract text from an EPUB file using ebooklib + BeautifulSoup."""
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        raise ImportError(
            "ebooklib is required for EPUB files. "
            "Install it: pip install ebooklib"
        )

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    texts: list[str] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        # Remove scripts / styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = _collapse_whitespace(text)
        if text.strip():
            texts.append(text.strip())

    return "\n\n".join(texts)


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple blank lines into a single one."""
    return re.sub(r"\n{3,}", "\n\n", text)


# ── Chapter detection ───────────────────────────────────────

# Common chapter heading patterns
_CHAPTER_PATTERNS = [
    re.compile(r"^chapter\s+\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^capitolo\s+\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#{1,3}\s+.+", re.MULTILINE),  # markdown headings
    re.compile(r"^\*{3,}$", re.MULTILINE),        # *** separator
    re.compile(r"^-{3,}$", re.MULTILINE),          # --- separator
]


def split_into_chapters(text: str) -> list[str]:
    """
    Try to split text at chapter boundaries.

    Falls back to splitting at double-newlines if no chapters are detected.
    Returns a list of chapter strings.
    """
    # Try each pattern
    for pattern in _CHAPTER_PATTERNS:
        matches = list(pattern.finditer(text))
        if len(matches) >= 2:
            chapters: list[str] = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                chunk = text[start:end].strip()
                if chunk:
                    chapters.append(chunk)
            return chapters

    # No chapter markers found — return the whole text as one chapter
    return [text.strip()]

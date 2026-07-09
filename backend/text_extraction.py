
from __future__ import annotations

import io

PDF_EXTENSIONS = {".pdf"}
KNOWN_TEXT_EXTENSIONS = {
    ".txt", ".log", ".md", ".json", ".csv", ".yaml", ".yml",
    ".out", ".err", ".xml", ".ini", ".conf", ".cfg",
}
KNOWN_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff",
    ".exe", ".dll", ".so", ".bin",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".mp3", ".mp4", ".mov", ".avi", ".wav",
    ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt",
    ".sqlite", ".db",
}

# If fewer than this fraction of decoded characters are printable/whitespace,
# treat it as binary content that happened to decode without erroring
# (Latin-1 can decode literally any byte sequence, so decoding alone isn't
# proof the file is actually text).
MIN_PRINTABLE_RATIO = 0.85


class FileExtractionError(Exception):
    """Raised when a file's text content can't be extracted."""


def extract_text(filename: str, content: bytes) -> str:
    ext = _extension(filename)

    if ext in KNOWN_BINARY_EXTENSIONS:
        raise FileExtractionError(
            f"'{ext}' files aren't supported. Upload a plain-text log (.txt, .log, .md, "
            f".json, .csv, ...) or a .pdf instead."
        )

    if ext in PDF_EXTENSIONS:
        return _extract_pdf(content)

    return _extract_plain_text(content)


def _extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _extract_plain_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding)
            _assert_looks_like_text(text)
            return text
        except UnicodeDecodeError:
            continue
    raise FileExtractionError(
        "Could not decode file as text (tried UTF-8 and Latin-1). "
        "If this is a binary format, only .pdf and plain-text formats are supported."
    )


def _assert_looks_like_text(text: str) -> None:
    """
    Latin-1 can decode any byte sequence without raising, so successful
    decoding alone doesn't prove a file is actually text (a PNG or ZIP
    will 'decode' just fine into garbage). Reject content that's mostly
    non-printable once decoded.
    """
    if not text.strip():
        return  # let the empty-content check further up the stack handle this
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    ratio = printable / len(text)
    if ratio < MIN_PRINTABLE_RATIO:
        raise FileExtractionError(
            "This file doesn't appear to contain readable text (it may be a binary "
            "format). Supported formats: plain-text logs (.txt, .log, .md, .json, "
            ".csv, ...) and .pdf."
        )


def _extract_pdf(content: bytes) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise FileExtractionError(
            "pdfplumber is not installed. Run: pip install pdfplumber"
        ) from exc

    text_parts: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
    except Exception as exc:  # noqa: BLE001
        raise FileExtractionError(f"Failed to read PDF: {exc}") from exc

    full_text = "\n\n".join(text_parts).strip()
    if not full_text:
        raise FileExtractionError(
            "No extractable text found in this PDF (it may be a scanned image without OCR)."
        )
    return full_text

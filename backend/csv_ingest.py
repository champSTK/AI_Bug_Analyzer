

from __future__ import annotations

import csv
import io


class CsvParseError(Exception):
    """Raised when the uploaded file can't be parsed as CSV."""


def parse_csv_rows(content: bytes) -> list[str]:
    """
    Parses CSV bytes into a list of raw-text blocks, one per data row
    (header row excluded). Empty rows are skipped entirely (not counted
    as failures — they're not a bug report to begin with).
    """
    text = _decode(content)

    # csv.Sniffer needs a sample; fall back to comma if sniffing fails
    # (e.g. single-column files, or a sample with no clear delimiter).
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel  # default: comma-delimited

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if reader.fieldnames is None:
        raise CsvParseError("Could not find a header row in this CSV file.")

    rows: list[str] = []
    for row in reader:
        block = _row_to_text(row)
        if block.strip():
            rows.append(block)

    if not rows:
        raise CsvParseError("No data rows found in this CSV file (only a header, or the file is empty).")

    return rows


def _decode(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise CsvParseError("Could not decode this file as text (tried UTF-8 and Latin-1).")


def _row_to_text(row: dict) -> str:
    """Flattens one CSV row into a 'Column: value' text block for Stage 1."""
    lines = []
    for key, value in row.items():
        if key is None:
            continue
        value_str = "" if value is None else str(value).strip()
        if value_str:
            lines.append(f"{key.strip()}: {value_str}")
    return "\n".join(lines)

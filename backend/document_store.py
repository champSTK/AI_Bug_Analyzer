

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"
DB_PATH = STORAGE_ROOT / "bug_analyzer.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS bug_reports (
    bug_id              TEXT PRIMARY KEY,
    source              TEXT NOT NULL,
    raw_text            TEXT NOT NULL,
    structured_json     TEXT NOT NULL,
    ingested_at         TEXT NOT NULL,
    validation_status   TEXT NOT NULL,
    extraction_attempts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bug_chunks (
    chunk_id    TEXT PRIMARY KEY,
    bug_id      TEXT NOT NULL REFERENCES bug_reports(bug_id),
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    char_count  INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bug_chunks_bug_id ON bug_chunks(bug_id);
"""


def _ensure_db() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def bulk_session() -> Iterator[sqlite3.Connection]:
    """
    One connection for an entire bulk import, instead of opening/closing/
    committing a new connection for every single row (which is what the
    per-call get_conn() does — fine for one-off calls, wasteful at
    thousands of rows). Caller is responsible for calling conn.commit()
    periodically (see save_bug_report_conn/save_chunks_conn) and a final
    commit happens automatically on a clean exit.
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# bug_reports
# ---------------------------------------------------------------------------

def save_bug_report(
    *,
    bug_id: str,
    source: str,
    raw_text: str,
    structured_json: str,
    ingested_at: str,
    validation_status: str,
    extraction_attempts: int,
) -> None:
    with get_conn() as conn:
        save_bug_report_conn(
            conn,
            bug_id=bug_id,
            source=source,
            raw_text=raw_text,
            structured_json=structured_json,
            ingested_at=ingested_at,
            validation_status=validation_status,
            extraction_attempts=extraction_attempts,
        )


def save_bug_report_conn(
    conn: sqlite3.Connection,
    *,
    bug_id: str,
    source: str,
    raw_text: str,
    structured_json: str,
    ingested_at: str,
    validation_status: str,
    extraction_attempts: int,
) -> None:
    """Same as save_bug_report(), but uses a connection you already opened
    (see bulk_session()) instead of opening/committing/closing its own."""
    conn.execute(
        """
        INSERT INTO bug_reports
            (bug_id, source, raw_text, structured_json, ingested_at,
             validation_status, extraction_attempts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            bug_id,
            source,
            raw_text,
            structured_json,
            ingested_at,
            validation_status,
            extraction_attempts,
        ),
    )


def get_bug_report(bug_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM bug_reports WHERE bug_id = ?", (bug_id,)
        ).fetchone()
        return dict(row) if row else None


def list_bug_reports() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bug_reports ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# bug_chunks
# ---------------------------------------------------------------------------

def save_chunks(bug_id: str, chunk_texts: list[str]) -> list[dict]:
    """Persists Stage-2 chunks for a bug report and returns the saved rows."""
    with get_conn() as conn:
        return save_chunks_conn(conn, bug_id, chunk_texts)


def save_chunks_conn(conn: sqlite3.Connection, bug_id: str, chunk_texts: list[str]) -> list[dict]:
    """Same as save_chunks(), but uses a connection you already opened
    (see bulk_session()) instead of opening/committing/closing its own."""
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for idx, text in enumerate(chunk_texts):
        chunk = {
            "chunk_id": str(uuid4()),
            "bug_id": bug_id,
            "chunk_index": idx,
            "chunk_text": text,
            "char_count": len(text),
            "created_at": now,
        }
        conn.execute(
            """
            INSERT INTO bug_chunks
                (chunk_id, bug_id, chunk_index, chunk_text, char_count, created_at)
            VALUES (:chunk_id, :bug_id, :chunk_index, :chunk_text, :char_count, :created_at)
            """,
            chunk,
        )
        rows.append(chunk)
    return rows


def get_chunks_for_bug(bug_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bug_chunks WHERE bug_id = ? ORDER BY chunk_index ASC",
            (bug_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_chunks() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bug_chunks ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def table_counts() -> dict:
    with get_conn() as conn:
        reports = conn.execute("SELECT COUNT(*) AS c FROM bug_reports").fetchone()["c"]
        chunks = conn.execute("SELECT COUNT(*) AS c FROM bug_chunks").fetchone()["c"]
        return {"bug_reports": reports, "bug_chunks": chunks}

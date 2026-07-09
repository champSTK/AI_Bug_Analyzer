

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import chunking
import csv_ingest
import document_store
import embeddings
import text_extraction
import vector_store
from groq_client import GroqExtractionError, GroqRateLimitError, extract_structured_report
from schemas import ImportSummaryResponse, RawIngestRequest, RowFailure, StructuredBugReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="AI Bug Analyzer — Stages 1-3 (local storage)",
    description=(
        "Raw log in -> Groq extraction -> local chunking -> local embeddings. "
        "PostgreSQL and Qdrant are simulated locally (SQLite + JSON) for now."
    ),
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_SOURCE = "dataset"
MAX_DATASET_ROWS = 20000          # hard safety ceiling on a single CSV import
MAX_FAILURES_RETURNED = 50        # keep the response light on huge imports
MAX_BUG_IDS_RETURNED = 200
PROGRESS_LOG_EVERY = 100


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "stage": "1-3", "storage": "local (sqlite + json)"}


# ---------------------------------------------------------------------------
# Core per-record pipeline: Stage 1 -> Stage 2 -> Stage 3
# ---------------------------------------------------------------------------

def _ingest_one_record(text: str, source: str) -> tuple[str | None, int, int, str | None]:
    """
    Runs Stages 1-3 for exactly one bug report (single-record path — each
    stage opens/commits its own storage write). Never raises — failures
    are reported back so a caller can record every single failure and
    keep going instead of aborting.

    Returns (bug_id_or_None, chunks_added, vectors_added, error_or_None).
    """
    try:
        structured, attempts = extract_structured_report(text)
    except GroqRateLimitError:
        return None, 0, 0, "Groq free-tier rate limit hit repeatedly for this record."
    except GroqExtractionError as exc:
        return None, 0, 0, str(exc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage 1 failed unexpectedly")
        return None, 0, 0, f"Stage 1 failed: {exc}"

    bug_id = str(uuid4())
    ingested_at = _now_iso()

    try:
        document_store.save_bug_report(
            bug_id=bug_id,
            source=source,
            raw_text=text,
            structured_json=structured.model_dump_json(),
            ingested_at=ingested_at,
            validation_status="VALID",
            extraction_attempts=attempts,
        )

        chunk_texts = chunking.chunk_structured_report(structured)
        chunk_rows = document_store.save_chunks(bug_id, chunk_texts)

        vectors = embeddings.embed_texts([c["chunk_text"] for c in chunk_rows])
        points = vector_store.upsert_points(bug_id, chunk_rows, vectors)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage 2/3 failed for bug_id=%s", bug_id)
        return None, 0, 0, f"Stage 2/3 failed: {exc}"

    return bug_id, len(chunk_rows), len(points), None


def _ingest_one_record_bulk(
    text: str,
    source: str,
    conn,
    vector_data: dict,
) -> tuple[str | None, int, int, str | None]:
    """
    Same as _ingest_one_record(), but for use inside a bulk import loop:
    writes go through a connection the caller already opened (bulk_session())
    and vector points accumulate in `vector_data` in memory rather than
    triggering a full read-modify-write of the vector store file per row.
    The caller commits `conn` and flushes `vector_data` periodically.
    """
    try:
        structured, attempts = extract_structured_report(text)
    except GroqRateLimitError:
        return None, 0, 0, "Groq free-tier rate limit hit repeatedly for this record."
    except GroqExtractionError as exc:
        return None, 0, 0, str(exc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage 1 failed unexpectedly")
        return None, 0, 0, f"Stage 1 failed: {exc}"

    bug_id = str(uuid4())
    ingested_at = _now_iso()

    try:
        document_store.save_bug_report_conn(
            conn,
            bug_id=bug_id,
            source=source,
            raw_text=text,
            structured_json=structured.model_dump_json(),
            ingested_at=ingested_at,
            validation_status="VALID",
            extraction_attempts=attempts,
        )

        chunk_texts = chunking.chunk_structured_report(structured)
        chunk_rows = document_store.save_chunks_conn(conn, bug_id, chunk_texts)

        vectors = embeddings.embed_texts([c["chunk_text"] for c in chunk_rows])
        points = vector_store.append_points(vector_data, bug_id, chunk_rows, vectors)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage 2/3 failed for bug_id=%s", bug_id)
        return None, 0, 0, f"Stage 2/3 failed: {exc}"

    return bug_id, len(chunk_rows), len(points), None


def _run_single(text: str, source: str) -> ImportSummaryResponse:
    start = time.perf_counter()

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="No text content to ingest.")

    bug_id, chunks_added, vectors_added, error = _ingest_one_record(text, source)
    elapsed = time.perf_counter() - start

    if error:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {error}")

    return ImportSummaryResponse(
        title="Bug Report Imported",
        source=source,
        successfully_imported=1,
        sqlite_reports_added=1,
        sqlite_chunks_added=chunks_added,
        vector_embeddings_added=vectors_added,
        failed_count=0,
        processing_time_seconds=round(elapsed, 2),
        bug_ids=[bug_id],
        failures=[],
    )


def _run_dataset_rows(rows: list[str], source: str) -> ImportSummaryResponse:
    start = time.perf_counter()

    truncated = len(rows) > MAX_DATASET_ROWS
    rows = rows[:MAX_DATASET_ROWS]

    reports_added = 0
    chunks_added_total = 0
    vectors_added_total = 0
    bug_ids: list[str] = []
    failures: list[RowFailure] = []

    total = len(rows)

    # One SQLite connection + one in-memory vector-store snapshot for the
    # WHOLE import, instead of a fresh connection and a full vector-store
    # file rewrite on every single row. That per-row pattern is what made
    # a 4000+ row CSV effectively O(n^2) and appear to hang. We commit /
    # flush periodically so progress is never more than one batch from
    # being durable, without paying the full-rewrite cost on every row.
    with document_store.bulk_session() as conn:
        vector_data = vector_store.load_for_bulk()

        for idx, row_text in enumerate(rows):
            bug_id, chunks_added, vectors_added, error = _ingest_one_record_bulk(
                row_text, source, conn, vector_data
            )

            if error:
                failures.append(
                    RowFailure(row_index=idx, preview=row_text[:120], error=error)
                )
            else:
                reports_added += 1
                chunks_added_total += chunks_added
                vectors_added_total += vectors_added
                if len(bug_ids) < MAX_BUG_IDS_RETURNED:
                    bug_ids.append(bug_id)

            is_last = (idx + 1) == total
            if (idx + 1) % PROGRESS_LOG_EVERY == 0 or is_last:
                conn.commit()
                vector_store.flush_bulk(vector_data)
                logger.info(
                    "Dataset import progress: %d/%d rows processed (%d succeeded, %d failed so far)",
                    idx + 1, total, reports_added, len(failures),
                )

    elapsed = time.perf_counter() - start

    if reports_added == 0 and failures:
        raise HTTPException(
            status_code=502,
            detail=f"All {total} row(s) failed extraction. First error: {failures[0].error}",
        )

    logger.info(
        "Dataset import complete: %d/%d succeeded, %d failed, %.2fs elapsed",
        reports_added, total, len(failures), elapsed,
    )

    return ImportSummaryResponse(
        title="Dataset Import Complete",
        source=source,
        successfully_imported=reports_added,
        sqlite_reports_added=reports_added,
        sqlite_chunks_added=chunks_added_total,
        vector_embeddings_added=vectors_added_total,
        failed_count=len(failures),
        processing_time_seconds=round(elapsed, 2),
        bug_ids=bug_ids,
        failures=failures[:MAX_FAILURES_RETURNED],
    )


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=ImportSummaryResponse)
def ingest(payload: RawIngestRequest) -> ImportSummaryResponse:
    """
    Pasted raw text is always treated as exactly one bug report,
    regardless of source — including source='dataset' (there's no CSV
    structure in pasted text to split on).
    """
    return _run_single(payload.raw_text, payload.source)


@app.post("/ingest/file", response_model=ImportSummaryResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source: str = Form(default="manual"),
) -> ImportSummaryResponse:
    """
    - source='dataset' + a .csv file  -> every row is its own bug report.
    - source='dataset' + any other file -> treated as one bug report
      (no row structure to split on outside CSV).
    - any other source                -> always one bug report.
    """
    content = await file.read()
    filename = file.filename or "upload"
    is_csv = filename.lower().endswith(".csv")

    if source == DATASET_SOURCE and is_csv:
        try:
            rows = csv_ingest.parse_csv_rows(content)
        except csv_ingest.CsvParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        logger.info("Dataset CSV parsed: %d row(s) detected in %s", len(rows), filename)
        return _run_dataset_rows(rows, source)

    try:
        text = text_extraction.extract_text(filename, content)
    except text_extraction.FileExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _run_single(text, source)


# ---------------------------------------------------------------------------
# Admin / viewer endpoints — power the separate "storage viewer" webpage.
# Read-only windows into the local SQLite (Postgres stand-in) and JSON
# (Qdrant stand-in) files.
# ---------------------------------------------------------------------------

@app.get("/admin/summary")
def admin_summary() -> dict:
    doc_counts = document_store.table_counts()
    vector_info = vector_store.get_collection_info()
    return {"document_store": doc_counts, "vector_store": vector_info}


@app.get("/admin/documents")
def admin_list_documents() -> list[dict]:
    """Simulated `SELECT * FROM bug_reports ORDER BY ingested_at DESC`."""
    rows = document_store.list_bug_reports()
    for row in rows:
        row["structured"] = StructuredBugReport.model_validate_json(
            row.pop("structured_json")
        ).model_dump()
    return rows


@app.get("/admin/documents/{bug_id}")
def admin_get_document(bug_id: str) -> dict:
    row = document_store.get_bug_report(bug_id)
    if row is None:
        raise HTTPException(status_code=404, detail="bug_id not found in document store.")
    row["structured"] = StructuredBugReport.model_validate_json(
        row.pop("structured_json")
    ).model_dump()
    row["chunks"] = document_store.get_chunks_for_bug(bug_id)
    return row


@app.get("/admin/chunks")
def admin_list_chunks() -> list[dict]:
    """Simulated `SELECT * FROM bug_chunks ORDER BY created_at DESC`."""
    return document_store.list_chunks()


@app.get("/admin/vectors")
def admin_list_vectors() -> list[dict]:
    """
    Simulated Qdrant `scroll()` over the local `bug_vectors` collection.
    Vectors are truncated to a short preview so the payload stays light;
    use /admin/vectors/{point_id} for the full float array.
    """
    points = vector_store.list_points()
    out = []
    for p in points:
        out.append(
            {
                "id": p["id"],
                "payload": p["payload"],
                "vector_preview": p["vector"][:8],
                "vector_dim": len(p["vector"]),
                "created_at": p["created_at"],
            }
        )
    return out


@app.get("/admin/vectors/{point_id}")
def admin_get_vector(point_id: str) -> dict:
    point = vector_store.get_point(point_id)
    if point is None:
        raise HTTPException(status_code=404, detail="Vector point not found.")
    return point

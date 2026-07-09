

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class RawIngestRequest(BaseModel):
    """What the client (Client App / CI-CD / JIRA webhook) sends us."""

    raw_text: str = Field(
        ...,
        min_length=1,
        description="Unformatted crash log, stack trace, or JIRA webhook body.",
    )
    source: str = Field(
        default="manual",
        description=(
            "Where this came from: 'manual', 'jira', 'ci-cd', 'slack' all mean "
            "'treat this submission as exactly one bug report'. 'dataset' is "
            "reserved for bulk CSV uploads via POST /ingest/file, where every "
            "row is its own bug report — pasted text with source='dataset' is "
            "still treated as a single report, since there's no CSV structure "
            "to split on."
        ),
    )


class EnvironmentInfo(BaseModel):
    os: Optional[str] = None
    runtime: Optional[str] = None
    version: Optional[str] = None


class ReportMetadata(BaseModel):
    title: Optional[str] = Field(default=None, max_length=150)
    severity: str = Field(default="MEDIUM")
    component: Optional[str] = None
    timestamp: Optional[str] = None

    @field_validator("severity")
    @classmethod
    def normalize_severity(cls, v: str) -> str:
        v = (v or "MEDIUM").upper()
        return v if v in {"LOW", "MEDIUM", "HIGH", "CRITICAL"} else "MEDIUM"


class StructuredBugReport(BaseModel):
    """
    The exact JSON shape the Stage-1 LLM extraction call must return.
    This is what gets JSON-mode-forced out of llama-3.1-8b-instant.
    """

    error_code: Optional[str] = Field(
        default=None, description="e.g. ECONNRESET, NullPointerException, 500"
    )
    message: str = Field(..., description="Human-readable summary of the error.")
    stack_trace: Optional[str] = Field(default=None)
    environment: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    metadata: ReportMetadata = Field(default_factory=ReportMetadata)


class StoredBugReport(BaseModel):
    """
    What actually gets written to the document store (Stage 1 output).
    Wraps the validated structured report with ingestion bookkeeping,
    ready to be picked up by Stage 2 (chunking).
    """

    bug_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    raw_text: str
    structured: StructuredBugReport
    ingested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    validation_status: str = "VALID"
    extraction_attempts: int = 1


class ChunkOut(BaseModel):
    """A single row of the (simulated) `bug_chunks` table."""

    chunk_id: str
    bug_id: str
    chunk_index: int
    chunk_text: str
    char_count: int
    created_at: str


class VectorPointOut(BaseModel):
    """A single point in the (simulated) Qdrant `bug_vectors` collection."""

    id: str
    bug_id: str
    chunk_id: str
    chunk_index: int
    text_preview: str
    vector_preview: list[float]
    vector_dim: int
    created_at: str


class IngestResponse(BaseModel):
    """Full pipeline result for a single bug report (used internally / for single-record imports)."""

    bug_id: str
    source: str
    structured: StructuredBugReport
    ingested_at: str
    validation_status: str
    extraction_attempts: int
    chunk_count: int
    chunks: list[ChunkOut]
    vector_count: int
    embedding_model: str
    embedding_dim: int


class RowFailure(BaseModel):
    """Records one row/record that failed Stage 1-3 processing without aborting the rest."""

    row_index: int
    preview: str
    error: str


class ImportSummaryResponse(BaseModel):
    """
    Returned by POST /ingest and POST /ingest/file. Every submission —
    a single pasted bug report or a multi-thousand-row dataset upload —
    reports through this same shape, so the frontend always shows one
    consistent import summary instead of rendering every structured
    record inline.
    """

    title: str
    source: str
    successfully_imported: int
    sqlite_reports_added: int
    sqlite_chunks_added: int
    vector_embeddings_added: int
    failed_count: int
    processing_time_seconds: float
    bug_ids: list[str] = Field(default_factory=list)
    failures: list[RowFailure] = Field(default_factory=list)

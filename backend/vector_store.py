

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np

STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"
VECTOR_STORE_FILE = STORAGE_ROOT / "vector_store.json"

COLLECTION_NAME = "bug_vectors"
VECTOR_SIZE = 384
DISTANCE_METRIC = "Cosine"

_lock = threading.Lock()


def _ensure_store() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    if not VECTOR_STORE_FILE.exists():
        VECTOR_STORE_FILE.write_text(
            json.dumps(
                {
                    "collection": COLLECTION_NAME,
                    "vector_size": VECTOR_SIZE,
                    "distance": DISTANCE_METRIC,
                    "points": [],
                },
                indent=2,
            )
        )


def _read() -> dict:
    _ensure_store()
    return json.loads(VECTOR_STORE_FILE.read_text())


def _write(data: dict) -> None:
    VECTOR_STORE_FILE.write_text(json.dumps(data, indent=2))


def upsert_points(
    bug_id: str,
    chunks: list[dict],
    vectors: list[list[float]],
) -> list[dict]:
    """
    Upserts one vector point per chunk for a SINGLE record. Reads and
    rewrites the whole file, which is fine for one-off calls but would be
    O(n^2) if called once per row in a large bulk import — use the
    load_for_bulk()/append_points()/flush_bulk() trio for that instead.
    """
    new_points = _build_points(bug_id, chunks, vectors)

    with _lock:
        data = _read()
        data["points"].extend(new_points)
        _write(data)

    return new_points


def _build_points(bug_id: str, chunks: list[dict], vectors: list[list[float]]) -> list[dict]:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must be the same length")

    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "id": str(uuid4()),
            "vector": vector,
            "payload": {
                "bug_id": bug_id,
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "text_preview": chunk["chunk_text"][:200],
            },
            "created_at": now,
        }
        for chunk, vector in zip(chunks, vectors)
    ]


# ---------------------------------------------------------------------------
# Bulk-import API — for datasets with many rows. Reads the store ONCE,
# accumulates every new point in memory, and writes ONCE (or in periodic
# batches), instead of doing a full read-modify-write of a growing file on
# every single row. That per-row read-modify-write pattern is what made
# large CSV imports effectively O(n^2) and appear to hang.
# ---------------------------------------------------------------------------

def load_for_bulk() -> dict:
    """Reads the store once at the start of a bulk import session."""
    with _lock:
        return _read()


def append_points(data: dict, bug_id: str, chunks: list[dict], vectors: list[list[float]]) -> list[dict]:
    """Appends points to the in-memory `data` dict only — no disk I/O."""
    new_points = _build_points(bug_id, chunks, vectors)
    data["points"].extend(new_points)
    return new_points


def flush_bulk(data: dict) -> None:
    """Writes the accumulated in-memory `data` dict to disk once."""
    with _lock:
        _write(data)


def list_points() -> list[dict]:
    data = _read()
    return data["points"]


def get_point(point_id: str) -> Optional[dict]:
    data = _read()
    for point in data["points"]:
        if point["id"] == point_id:
            return point
    return None


def get_collection_info() -> dict:
    data = _read()
    return {
        "collection": data["collection"],
        "vector_size": data["vector_size"],
        "distance": data["distance"],
        "points_count": len(data["points"]),
    }


def search(query_vector: list[float], top_k: int = 3) -> list[dict]:
    """
    Cosine-similarity search over the local point set — the same
    operation a real Qdrant ANN search performs, just brute-forced
    with numpy since this is a local stand-in. Vectors are expected
    to already be normalized (bge models: normalize_embeddings=True),
    so cosine similarity reduces to a dot product.
    """
    data = _read()
    points = data["points"]
    if not points:
        return []

    matrix = np.array([p["vector"] for p in points])
    query = np.array(query_vector)
    scores = matrix @ query

    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for i in top_idx:
        point = dict(points[int(i)])
        point["score"] = float(scores[int(i)])
        results.append(point)
    return results

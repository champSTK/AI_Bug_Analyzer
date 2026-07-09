
from __future__ import annotations

import threading

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_model = None
_model_lock = threading.Lock()


def get_model():
    """Lazily loads and caches the embedding model (loaded once per process)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked locking
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generates normalized embeddings for a batch of text chunks.
    Normalized vectors mean cosine similarity == dot product, which is
    what vector_store.search() relies on.
    """
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()

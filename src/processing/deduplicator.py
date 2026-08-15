"""
src/processing/deduplicator.py

HuggingFace's real, scoped role in this project: uses a small, local
sentence-embedding model (all-MiniLM-L6-v2, see config.py) to detect
and remove near-duplicate chunks before they reach Groq for
summarization.

This does NOT generate any part of the summary itself -- Groq does
all of that. This module only decides which chunks are worth sending.
"""

from functools import lru_cache
from typing import List
import threading
import contextlib

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

from src.config import (
    DEDUP_SIMILARITY_THRESHOLD,
    HUGGINGFACE_EMBEDDING_MODEL,
    HUGGINGFACE_EMBEDDING_TIMEOUT_SECONDS,
)
from src.logger import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def _model_loading_timeout(seconds: int):
    """
    Context manager for enforcing timeout on HuggingFace model loading.
    Cross-platform implementation using threading.Timer.

    NOTE: This is a best-effort timeout. For blocking I/O operations
    (like downloading from HuggingFace hub), Python's threading cannot
    truly interrupt the operation. However, we can:
    1. Set a timeout and log a warning if exceeded
    2. Return control to Streamlit, which has its own 5-minute session timeout
    3. Document this limitation for deployment teams

    For production systems that need guaranteed timeouts, use:
    - uvloop with asyncio
    - A separate service with process-level timeout (e.g., timeout(1) on Linux)
    - A thread pool with explicit cancellation
    """

    timeout_event = threading.Event()

    def _timeout_callback():
        """Called when timeout expires."""
        timeout_event.set()
        logger.warning(
            "HuggingFace model loading exceeded %d seconds. "
            "This may indicate a slow network connection. "
            "Streamlit will timeout the session after 5 minutes.",
            seconds,
        )

    timer = threading.Timer(seconds, _timeout_callback)
    timer.daemon = True
    timer.start()

    try:
        yield timeout_event
    finally:
        timer.cancel()


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """
    Loads the HuggingFace embedding model once per process and reuses
    it for the app's lifetime. Loading is not free -- first run
    downloads the model from HuggingFace's hub (~90MB) and caches it
    locally; every run after that loads from local cache. lru_cache
    gives a simple, framework-agnostic singleton -- no coupling to
    st.cache_resource, so this module works identically from
    Streamlit, a test, or a plain script.

    TIMEOUT BEHAVIOR:
    - If model download takes longer than HUGGINGFACE_EMBEDDING_TIMEOUT_SECONDS,
      a warning is logged, but the operation is NOT forcefully stopped
      (Python threading cannot truly interrupt blocking I/O).
    - Streamlit Cloud has a 5-minute session timeout, so extremely slow
      connections will fail there.
    - Local development will eventually complete if network is working,
      even if slow.

    MEMORY LIFECYCLE:
    - On Streamlit Cloud with persistent workers, the model stays in memory
      across sessions. This is intentional for performance (reusing ~90MB model
      is cheaper than redownloading), but creates a memory footprint.
    - For long-running deployments with many concurrent users, consider using
      Streamlit's st.cache_resource(ttl=3600) for TTL-based eviction (future
      enhancement).
    """
    logger.info("Loading HuggingFace embedding model: %s", HUGGINGFACE_EMBEDDING_MODEL)

    with _model_loading_timeout(HUGGINGFACE_EMBEDDING_TIMEOUT_SECONDS) as timeout_event:
        try:
            model = SentenceTransformer(HUGGINGFACE_EMBEDDING_MODEL)

            if timeout_event.is_set():
                logger.warning(
                    "Model loaded successfully, but took longer than %d seconds.",
                    HUGGINGFACE_EMBEDDING_TIMEOUT_SECONDS,
                )
            else:
                logger.info("Successfully loaded embedding model.")

            return model

        except Exception as exc:
            # Re-raise with context about the failure
            from src.summarization.base_strategy import SummarizationError

            if timeout_event.is_set():
                error_msg = (
                    "The deduplication step exceeded the timeout while loading the "
                    f"embedding model (>{HUGGINGFACE_EMBEDDING_TIMEOUT_SECONDS}s). "
                    "This usually means a slow network connection to HuggingFace hub. "
                    "Please try again, or disable deduplication in a future version."
                )
            else:
                error_msg = (
                    "Failed to load the embedding model for deduplication. "
                    "This usually indicates a network issue or corrupted local cache. "
                    "Try deleting ~/.cache/huggingface and trying again."
                )

            raise SummarizationError(error_msg, cause=exc) from exc


def deduplicate_chunks(
    chunks: List[Document],
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> List[Document]:
    """
    Removes near-duplicate chunks, keeping the first occurrence of each
    group of similar chunks and dropping later ones. Preserves the
    original relative order of the chunks that are kept.

    Matters most for Map-Reduce, where every surviving chunk costs one
    Groq call -- YouTube transcripts especially tend to repeat intros,
    sponsor reads, or restated points, and dropping near-duplicates
    before summarization avoids paying for (and slightly diluting the
    final summary with) redundant content.

    Uses a greedy O(n^2) comparison: each chunk is compared against
    every already-kept chunk. This is fine for the chunk counts this
    project deals with (typically tens, occasionally low hundreds) --
    not something tested or claimed to scale beyond that.
    """
    if len(chunks) <= 1:
        return chunks

    model = _get_embedding_model()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    kept_chunks: List[Document] = []
    kept_embeddings: List[np.ndarray] = []

    for chunk, embedding in zip(chunks, embeddings):
        if kept_embeddings:
            similarities = util.cos_sim(embedding, np.vstack(kept_embeddings))[0]
            max_similarity = float(similarities.max())
        else:
            max_similarity = 0.0

        if max_similarity >= similarity_threshold:
            logger.info(
                "Dropping near-duplicate chunk_index=%s (similarity=%.3f >= threshold=%.2f)",
                chunk.metadata.get("chunk_index"),
                max_similarity,
                similarity_threshold,
            )
            continue

        kept_chunks.append(chunk)
        kept_embeddings.append(embedding)

    logger.info(
        "Deduplication: %d chunks -> %d chunks (%d dropped, threshold=%.2f)",
        len(chunks),
        len(kept_chunks),
        len(chunks) - len(kept_chunks),
        similarity_threshold,
    )

    return kept_chunks

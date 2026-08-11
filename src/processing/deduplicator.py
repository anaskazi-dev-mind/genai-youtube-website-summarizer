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

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

from src.config import DEDUP_SIMILARITY_THRESHOLD, HUGGINGFACE_EMBEDDING_MODEL
from src.logger import get_logger

logger = get_logger(__name__)


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
    """
    logger.info("Loading HuggingFace embedding model: %s", HUGGINGFACE_EMBEDDING_MODEL)
    return SentenceTransformer(HUGGINGFACE_EMBEDDING_MODEL)


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

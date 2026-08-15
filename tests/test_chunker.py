"""
tests/test_chunker.py

Tests for src/processing/chunker.py.

CHUNK_SIZE and CHUNK_OVERLAP are monkeypatched to small values for
these tests, so multi-chunk behavior can be tested against short
strings instead of needing multi-kilobyte documents.

Important: chunker.py does `from src.config import CHUNK_SIZE`, which
binds a local name inside chunker.py's own namespace at import time.
Patching src.config.CHUNK_SIZE would NOT affect chunker.py's already-
bound copy -- so these tests patch src.processing.chunker.CHUNK_SIZE
directly, at the point where it's actually used.

IMPORTANT: Tests pass enforce_max_chunks=False to split_documents() to
bypass the MAX_CHUNKS_PER_CONTENT production safeguard. This allows unit
tests to focus on chunking behavior in isolation. The safeguard itself
is tested separately in test_max_chunks_limit_enforced_in_production.
"""

import pytest
from langchain_core.documents import Document

from src.processing import chunker
from src.processing.chunker import estimate_token_count, split_documents
from src.summarization.base_strategy import SummarizationError


def _long_document(source_url="https://example.com/a", num_sentences=30):
    text = " ".join(
        f"This is sentence number {i} in a long test document."
        for i in range(num_sentences)
    )
    return Document(
        page_content=text, metadata={"source_type": "website", "source_url": source_url}
    )


@pytest.fixture
def small_chunks(monkeypatch):
    """Overrides chunk size/overlap to small values, scoped to chunker.py's namespace."""
    monkeypatch.setattr(chunker, "CHUNK_SIZE", 100)
    monkeypatch.setattr(chunker, "CHUNK_OVERLAP", 20)


# ---------------------------------------------------------------------------
# split_documents -- chunking behavior (with enforce_max_chunks=False)
# ---------------------------------------------------------------------------


def test_split_documents_produces_multiple_chunks_for_long_text(small_chunks):
    document = _long_document(num_sentences=30)
    # Pass enforce_max_chunks=False to test chunking behavior in isolation
    # without triggering the MAX_CHUNKS_PER_CONTENT production safeguard.
    chunks = split_documents([document], enforce_max_chunks=False)

    assert len(chunks) > 1
    for chunk in chunks:
        # Some slack allowed: the splitter snaps to word/sentence
        # boundaries rather than cutting at an exact character count.
        assert len(chunk.page_content) <= 100 + 20


def test_split_documents_returns_single_chunk_for_short_text(small_chunks):
    document = Document(
        page_content="Short text that fits in one chunk.",
        metadata={"source_type": "website", "source_url": "https://x.com"},
    )
    chunks = split_documents([document], enforce_max_chunks=False)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text that fits in one chunk."


def test_split_documents_preserves_original_metadata(small_chunks):
    document = _long_document(
        source_url="https://example.com/article", num_sentences=20
    )
    chunks = split_documents([document], enforce_max_chunks=False)

    for chunk in chunks:
        assert chunk.metadata["source_type"] == "website"
        assert chunk.metadata["source_url"] == "https://example.com/article"


def test_split_documents_adds_sequential_chunk_index(small_chunks):
    document = _long_document(num_sentences=30)
    chunks = split_documents([document], enforce_max_chunks=False)

    indices = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indices == list(range(len(chunks)))


def test_split_documents_adds_increasing_start_index(small_chunks):
    document = _long_document(num_sentences=30)
    chunks = split_documents([document], enforce_max_chunks=False)

    start_indices = [chunk.metadata["start_index"] for chunk in chunks]
    assert start_indices == sorted(start_indices)
    assert start_indices[0] == 0


def test_split_documents_overlap_produces_shared_content_between_consecutive_chunks(
    small_chunks,
):
    # Proves overlap isn't just a number we pass through unused -- the
    # tail of one chunk should genuinely reappear in the next chunk.
    document = _long_document(num_sentences=30)
    chunks = split_documents([document], enforce_max_chunks=False)

    assert len(chunks) >= 2
    overlap_sample = chunks[0].page_content[-10:]
    assert overlap_sample in chunks[1].page_content


def test_split_documents_scopes_chunk_index_per_source(small_chunks):
    doc_a = _long_document(source_url="https://a.com", num_sentences=20)
    doc_b = _long_document(source_url="https://b.com", num_sentences=20)

    chunks = split_documents([doc_a, doc_b], enforce_max_chunks=False)

    a_indices = [
        c.metadata["chunk_index"]
        for c in chunks
        if c.metadata["source_url"] == "https://a.com"
    ]
    b_indices = [
        c.metadata["chunk_index"]
        for c in chunks
        if c.metadata["source_url"] == "https://b.com"
    ]

    assert a_indices == list(range(len(a_indices)))
    assert b_indices == list(range(len(b_indices)))


def test_split_documents_handles_empty_list():
    assert split_documents([]) == []


# ---------------------------------------------------------------------------
# MAX_CHUNKS_PER_CONTENT safeguard (production mode enforcement)
# ---------------------------------------------------------------------------


def test_max_chunks_limit_enforced_by_default(small_chunks, monkeypatch):
    """
    Verifies that enforce_max_chunks defaults to True, and the MAX_CHUNKS
    safeguard is active in production code. This prevents runaway API costs
    on extremely long content.
    """
    # Create content that will produce many chunks with small_chunks settings
    document = _long_document(num_sentences=100)

    # Monkeypatch MAX_CHUNKS_PER_CONTENT to a small value so we definitely
    # exceed it with this long document
    monkeypatch.setattr(chunker, "MAX_CHUNKS_PER_CONTENT", 5)

    # With enforce_max_chunks=True (default), should raise SummarizationError
    with pytest.raises(SummarizationError, match="too long to process"):
        split_documents([document])  # enforce_max_chunks=True by default


def test_max_chunks_limit_can_be_disabled_for_testing(small_chunks, monkeypatch):
    """
    Verifies that tests can disable the MAX_CHUNKS safeguard by passing
    enforce_max_chunks=False. This allows unit tests to focus on chunking
    behavior without production limits interfering.
    """
    document = _long_document(num_sentences=100)

    # Set a very low limit that would normally fail
    monkeypatch.setattr(chunker, "MAX_CHUNKS_PER_CONTENT", 5)

    # With enforce_max_chunks=False, should succeed even though we exceed limit
    chunks = split_documents([document], enforce_max_chunks=False)
    assert len(chunks) > 5  # Proves we actually exceeded the limit


def test_max_chunks_limit_with_realistic_production_settings(monkeypatch):
    """
    Integration test: verifies the safeguard works with realistic config
    values (CHUNK_SIZE=6000, MAX_CHUNKS_PER_CONTENT=15).

    This test creates a document that, when chunked with production settings,
    produces enough chunks to trigger the safeguard.
    """
    # Temporarily set production config values for this test
    monkeypatch.setattr(chunker, "CHUNK_SIZE", 6000)
    monkeypatch.setattr(chunker, "CHUNK_OVERLAP", 500)
    monkeypatch.setattr(chunker, "MAX_CHUNKS_PER_CONTENT", 15)

    # Create a very long document: ~200K chars = ~33K tokens
    # With CHUNK_SIZE=6000, this produces ~35 chunks
    very_long_text = " ".join([f"This is sentence {i}. " for i in range(5000)])
    document = Document(
        page_content=very_long_text,
        metadata={"source_type": "website", "source_url": "https://example.com/long"},
    )

    # Should raise with enforce_max_chunks=True (default)
    with pytest.raises(SummarizationError, match="too long to process"):
        split_documents([document])

    # Should succeed with enforce_max_chunks=False
    chunks = split_documents([document], enforce_max_chunks=False)
    assert len(chunks) > 15  # Proves we exceeded the configured limit


# ---------------------------------------------------------------------------
# estimate_token_count
# ---------------------------------------------------------------------------


def test_estimate_token_count_uses_four_chars_per_token_heuristic():
    assert estimate_token_count("a" * 400) == 100


def test_estimate_token_count_handles_empty_string():
    assert estimate_token_count("") == 0

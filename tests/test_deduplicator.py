"""
tests/test_deduplicator.py

Tests for src/processing/deduplicator.py.

SentenceTransformer is fully mocked -- no real model is downloaded or
loaded. We control exactly which embedding vectors .encode() returns,
so cosine-similarity outcomes are fully deterministic: we can
construct precise "duplicate" and "distinct" cases by hand rather than
relying on a real model's actual output.

_get_embedding_model() uses functools.lru_cache(maxsize=1), so the
FIRST call in a test session would normally be reused by every test
after it. clear_model_cache (autouse) clears that cache before and
after every test, so each test gets its own freshly-mocked model.
"""

import numpy as np
import pytest
from langchain_core.documents import Document

from src.config import HUGGINGFACE_EMBEDDING_MODEL
from src.processing import deduplicator
from src.processing.deduplicator import deduplicate_chunks


@pytest.fixture(autouse=True)
def clear_model_cache():
    deduplicator._get_embedding_model.cache_clear()
    yield
    deduplicator._get_embedding_model.cache_clear()


def _make_chunk(text, chunk_index):
    return Document(
        page_content=text,
        metadata={
            "source_type": "youtube",
            "source_url": "https://x.com",
            "chunk_index": chunk_index,
        },
    )


def _mock_model_with_embeddings(mocker, embeddings):
    """
    Patches SentenceTransformer so _get_embedding_model() returns a
    fake model whose .encode() call returns exactly these embeddings,
    in order, regardless of the input text.
    """
    fake_model = mocker.MagicMock()
    fake_model.encode.return_value = np.array(embeddings, dtype=np.float32)
    mock_class = mocker.patch("src.processing.deduplicator.SentenceTransformer")
    mock_class.return_value = fake_model
    return fake_model, mock_class


# ---------------------------------------------------------------------------
# Trivial cases -- no model call needed
# ---------------------------------------------------------------------------


def test_empty_list_returns_empty_and_never_loads_model(mocker):
    mock_class = mocker.patch("src.processing.deduplicator.SentenceTransformer")
    assert deduplicate_chunks([]) == []
    mock_class.assert_not_called()


def test_single_chunk_returned_unchanged_and_never_loads_model(mocker):
    mock_class = mocker.patch("src.processing.deduplicator.SentenceTransformer")
    chunk = _make_chunk("Only chunk.", 0)

    result = deduplicate_chunks([chunk])

    assert result == [chunk]
    mock_class.assert_not_called()


# ---------------------------------------------------------------------------
# Similarity-based dropping
# ---------------------------------------------------------------------------


def test_distinct_chunks_are_all_kept(mocker):
    # Orthogonal vectors -> cosine similarity of 0 between every pair.
    _mock_model_with_embeddings(
        mocker, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    chunks = [_make_chunk(f"Chunk {i}", i) for i in range(3)]

    result = deduplicate_chunks(chunks)

    assert result == chunks


def test_identical_chunks_second_one_dropped(mocker):
    # Identical vectors -> cosine similarity of exactly 1.0.
    _mock_model_with_embeddings(mocker, [[1.0, 0.0], [1.0, 0.0]])
    chunk_a = _make_chunk("Repeated intro.", 0)
    chunk_b = _make_chunk("Repeated intro.", 1)

    result = deduplicate_chunks([chunk_a, chunk_b])

    assert result == [chunk_a]


def test_non_adjacent_duplicate_is_still_caught(mocker):
    # chunk 0 and chunk 2 are identical; chunk 1 sits between them and
    # is distinct. Proves each chunk is compared against EVERY kept
    # chunk so far, not just the immediately preceding one.
    _mock_model_with_embeddings(
        mocker, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    chunk_0 = _make_chunk("Sponsor read.", 0)
    chunk_1 = _make_chunk("Unrelated content.", 1)
    chunk_2 = _make_chunk("Sponsor read.", 2)

    result = deduplicate_chunks([chunk_0, chunk_1, chunk_2])

    assert result == [chunk_0, chunk_1]


def test_similarity_threshold_is_configurable(mocker):
    # Constructed so cosine similarity between the two vectors is
    # exactly 0.95 (both are unit vectors by construction: 0.95^2 +
    # 0.3122^2 ≈ 1.0).
    _mock_model_with_embeddings(mocker, [[1.0, 0.0], [0.95, 0.31224989991992]])
    chunks = [_make_chunk("Text A", 0), _make_chunk("Text B", 1)]

    # Default threshold (0.92) -- 0.95 similarity counts as duplicate.
    assert deduplicate_chunks(chunks) == [chunks[0]]

    # Stricter custom threshold -- 0.95 similarity is now NOT enough
    # to count as duplicate.
    assert deduplicate_chunks(chunks, similarity_threshold=0.99) == chunks


# ---------------------------------------------------------------------------
# Model loading behavior
# ---------------------------------------------------------------------------


def test_model_is_loaded_with_configured_model_name(mocker):
    _, mock_class = _mock_model_with_embeddings(mocker, [[1.0, 0.0], [0.0, 1.0]])
    deduplicate_chunks([_make_chunk("A", 0), _make_chunk("B", 1)])

    mock_class.assert_called_once_with(HUGGINGFACE_EMBEDDING_MODEL)


def test_model_is_loaded_only_once_across_multiple_calls(mocker):
    _, mock_class = _mock_model_with_embeddings(mocker, [[1.0, 0.0], [0.0, 1.0]])
    chunks = [_make_chunk("A", 0), _make_chunk("B", 1)]

    deduplicate_chunks(chunks)
    deduplicate_chunks(chunks)

    # Proves lru_cache is doing its job: the constructor should only
    # run once even though deduplicate_chunks was called twice.
    assert mock_class.call_count == 1


def test_encode_is_called_once_with_all_chunk_texts_in_order(mocker):
    fake_model, _ = _mock_model_with_embeddings(mocker, [[1.0, 0.0], [0.0, 1.0]])
    chunks = [_make_chunk("First chunk text.", 0), _make_chunk("Second chunk text.", 1)]

    deduplicate_chunks(chunks)

    fake_model.encode.assert_called_once()
    called_texts = fake_model.encode.call_args.args[0]
    assert called_texts == ["First chunk text.", "Second chunk text."]

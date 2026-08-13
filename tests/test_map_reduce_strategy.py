"""
tests/test_map_reduce_strategy.py

Tests for src/summarization/map_reduce_strategy.py.

Two testing styles are used deliberately:
1. Real chain composition (via RunnableLambda fakes standing in for
   ChatGroq) -- proves prompt -> llm -> parser actually wires
   correctly for BOTH the map step (batch) and reduce step (invoke),
   and that chunk order is preserved end-to-end through the real code
   path, not just LangChain's own batch() (already verified directly
   in the conversation, see map_reduce_strategy.py's docstring).
2. Mocking safe_batch/safe_invoke directly -- isolates config
   pass-through (max_concurrency) and the reduce step's oversized-
   input guard without needing to construct a full fake chain for
   those specific claims.

get_llm() is called TWICE in MapReduceStrategy.summarize() (once to
build the map chain, once for the reduce chain) -- tests using
RunnableLambda fakes patch get_llm with side_effect=[map_llm,
reduce_llm] so each call returns the right fake.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.config import MAP_REDUCE_MAX_CONCURRENCY, STUFF_STRATEGY_MAX_ESTIMATED_TOKENS
from src.summarization.base_strategy import SummarizationError, SummaryResult
from src.summarization.map_reduce_strategy import MapReduceStrategy


def _make_chunk(text, source_url="https://example.com/a", chunk_index=0):
    return Document(
        page_content=text,
        metadata={
            "source_type": "website",
            "source_url": source_url,
            "title": "Test Article",
            "chunk_index": chunk_index,
        },
    )


def _fake_map_llm():
    """
    Stands in for ChatGroq during the map step. Echoes the chunk text
    back prefixed with "Summary of:", so tests can trace which
    original chunk each intermediate summary came from.
    """

    def _run(prompt_value):
        human_content = prompt_value.to_messages()[-1].content
        return AIMessage(content=f"Summary of: {human_content}")

    return RunnableLambda(_run)


def _fake_reduce_llm(fixed_text, captured_calls=None):
    """Stands in for ChatGroq during the reduce step."""

    def _run(prompt_value):
        if captured_calls is not None:
            captured_calls.append(prompt_value)
        return AIMessage(content=fixed_text)

    return RunnableLambda(_run)


@pytest.fixture
def strategy():
    return MapReduceStrategy()


# ---------------------------------------------------------------------------
# Success path -- real chain composition via RunnableLambda fakes
# ---------------------------------------------------------------------------


def test_summarize_returns_summary_result_with_correct_fields(mocker, strategy):
    mocker.patch(
        "src.summarization.map_reduce_strategy.get_llm",
        side_effect=[_fake_map_llm(), _fake_reduce_llm("## Title\nFinal Summary")],
    )
    chunks = [_make_chunk("Chunk one text."), _make_chunk("Chunk two text.")]

    result = strategy.summarize(chunks)

    assert isinstance(result, SummaryResult)
    assert result.content == "## Title\nFinal Summary"
    assert result.strategy == "map_reduce"
    assert result.chunk_count == 2
    assert result.source_metadata["source_type"] == "website"


def test_reduce_step_receives_numbered_summaries_in_original_order(mocker, strategy):
    captured = []
    mocker.patch(
        "src.summarization.map_reduce_strategy.get_llm",
        side_effect=[
            _fake_map_llm(),
            _fake_reduce_llm("final", captured_calls=captured),
        ],
    )
    chunks = [
        _make_chunk("Alpha content.", chunk_index=0),
        _make_chunk("Beta content.", chunk_index=1),
        _make_chunk("Gamma content.", chunk_index=2),
    ]

    strategy.summarize(chunks)

    assert len(captured) == 1
    reduce_input = captured[0].to_messages()[-1].content

    # Proves each chunk's map-summary landed in the CORRECT section
    # number, in original order -- not just that all three appear
    # somewhere in the combined text.
    section_1 = reduce_input.split("[Section 2]")[0]
    section_2 = reduce_input.split("[Section 2]")[1].split("[Section 3]")[0]
    section_3 = reduce_input.split("[Section 3]")[1]

    assert "Alpha content." in section_1
    assert "Beta content." in section_2
    assert "Gamma content." in section_3


def test_map_step_summarizes_each_chunk_independently(mocker, strategy):
    # Each chunk's map-summary should be based on THAT chunk alone --
    # proves the map step doesn't accidentally leak content across
    # chunks (e.g. by reusing stale state between batch items).
    captured = []
    mocker.patch(
        "src.summarization.map_reduce_strategy.get_llm",
        side_effect=[
            _fake_map_llm(),
            _fake_reduce_llm("final", captured_calls=captured),
        ],
    )
    chunks = [_make_chunk("Unique chunk A."), _make_chunk("Unique chunk B.")]

    strategy.summarize(chunks)

    reduce_input = captured[0].to_messages()[-1].content
    assert "Summary of: Unique chunk A." in reduce_input
    assert "Summary of: Unique chunk B." in reduce_input


# ---------------------------------------------------------------------------
# Config pass-through -- mocking safe_batch/safe_invoke directly
# ---------------------------------------------------------------------------


def test_map_step_uses_configured_max_concurrency(mocker, strategy):
    mocker.patch("src.summarization.map_reduce_strategy.get_llm")
    mock_safe_batch = mocker.patch(
        "src.summarization.map_reduce_strategy.safe_batch",
        return_value=["summary A", "summary B"],
    )
    mocker.patch(
        "src.summarization.map_reduce_strategy.safe_invoke",
        return_value="final summary",
    )
    chunks = [_make_chunk("A"), _make_chunk("B")]

    strategy.summarize(chunks)

    _, kwargs = mock_safe_batch.call_args
    assert kwargs["max_concurrency"] == MAP_REDUCE_MAX_CONCURRENCY


def test_chunk_count_matches_number_of_input_documents(mocker, strategy):
    mocker.patch("src.summarization.map_reduce_strategy.get_llm")
    mocker.patch(
        "src.summarization.map_reduce_strategy.safe_batch",
        return_value=["s1", "s2", "s3", "s4"],
    )
    mocker.patch(
        "src.summarization.map_reduce_strategy.safe_invoke",
        return_value="final",
    )
    chunks = [_make_chunk(f"Chunk {i}.", chunk_index=i) for i in range(4)]

    result = strategy.summarize(chunks)

    assert result.chunk_count == 4


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


def test_summarize_raises_on_empty_document_list(strategy):
    with pytest.raises(SummarizationError, match="no content"):
        strategy.summarize([])


def test_reduce_step_rejects_oversized_combined_summaries_without_calling_llm(
    mocker, strategy
):
    # Simulates the honest edge case documented in
    # map_reduce_strategy.py: even after per-chunk summarization, the
    # combined intermediate summaries could theoretically still be too
    # large for a single reduce call.
    huge_summary = "x" * ((STUFF_STRATEGY_MAX_ESTIMATED_TOKENS + 10_000) * 4)
    mocker.patch("src.summarization.map_reduce_strategy.get_llm")
    mocker.patch(
        "src.summarization.map_reduce_strategy.safe_batch",
        return_value=[huge_summary],
    )
    mock_safe_invoke = mocker.patch("src.summarization.map_reduce_strategy.safe_invoke")
    chunks = [_make_chunk("Some chunk.")]

    with pytest.raises(SummarizationError, match="too large for a single reduce call"):
        strategy.summarize(chunks)

    # Proves the guard prevents the reduce call entirely, rather than
    # sending an oversized request and hoping Groq handles it.
    mock_safe_invoke.assert_not_called()

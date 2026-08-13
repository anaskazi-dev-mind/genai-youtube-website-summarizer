"""
tests/test_stuff_strategy.py

Tests for src/summarization/stuff_strategy.py.

get_llm() is patched to return a RunnableLambda standing in for
ChatGroq -- NOT a plain MagicMock. This matters: MagicMock doesn't
implement LangChain's Runnable protocol, so `STUFF_PROMPT | mock_llm`
would fail or behave unpredictably. RunnableLambda IS a real Runnable,
so the actual LCEL pipe composition (prompt | llm | StrOutputParser())
runs for real in these tests -- only the "LLM" at the end is fake.
Verified directly that a RunnableLambda receives the real
ChatPromptValue LangChain produces, so we can inspect exactly what
would have been sent to Groq.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.config import STUFF_STRATEGY_MAX_ESTIMATED_TOKENS
from src.summarization.base_strategy import SummarizationError, SummaryResult
from src.summarization.stuff_strategy import StuffStrategy


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


def _fake_llm_returning(fixed_text, captured_calls=None):
    """
    Stands in for ChatGroq: a real Runnable (RunnableLambda) that
    returns a fixed AIMessage given a ChatPromptValue. If
    captured_calls is provided, each received ChatPromptValue is
    recorded so tests can assert on the exact content Groq would
    have received.
    """

    def _run(prompt_value):
        if captured_calls is not None:
            captured_calls.append(prompt_value)
        return AIMessage(content=fixed_text)

    return RunnableLambda(_run)


@pytest.fixture
def strategy():
    return StuffStrategy()


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_summarize_returns_summary_result_with_correct_fields(mocker, strategy):
    mocker.patch(
        "src.summarization.stuff_strategy.get_llm",
        return_value=_fake_llm_returning("## Title\nTest Summary"),
    )
    chunks = [_make_chunk("First part of the content.")]

    result = strategy.summarize(chunks)

    assert isinstance(result, SummaryResult)
    assert result.content == "## Title\nTest Summary"
    assert result.strategy == "stuff"
    assert result.chunk_count == 1
    assert result.source_metadata["source_type"] == "website"
    assert result.source_metadata["source_url"] == "https://example.com/a"


def test_summarize_combines_all_chunks_into_a_single_llm_call(mocker, strategy):
    captured = []
    mocker.patch(
        "src.summarization.stuff_strategy.get_llm",
        return_value=_fake_llm_returning("summary", captured_calls=captured),
    )
    chunks = [
        _make_chunk("First chunk text.", chunk_index=0),
        _make_chunk("Second chunk text.", chunk_index=1),
    ]

    strategy.summarize(chunks)

    # Exactly one call proves Stuff really does everything in ONE
    # LLM call, not one call per chunk.
    assert len(captured) == 1
    human_message = captured[0].to_messages()[-1]
    assert "First chunk text." in human_message.content
    assert "Second chunk text." in human_message.content


def test_summarize_reports_correct_chunk_count(mocker, strategy):
    mocker.patch(
        "src.summarization.stuff_strategy.get_llm",
        return_value=_fake_llm_returning("summary"),
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


def test_summarize_rejects_oversized_content_without_calling_groq(mocker, strategy):
    mock_get_llm = mocker.patch("src.summarization.stuff_strategy.get_llm")
    # ~4 chars/token estimate -> comfortably exceeds the token limit.
    oversized_text = "x" * ((STUFF_STRATEGY_MAX_ESTIMATED_TOKENS + 10_000) * 4)
    chunks = [_make_chunk(oversized_text)]

    with pytest.raises(SummarizationError, match="too long for the Stuff strategy"):
        strategy.summarize(chunks)

    # Proves the size check happens BEFORE get_llm/Groq is ever touched
    # -- this is the whole point of the proactive check.
    mock_get_llm.assert_not_called()


def test_oversized_content_error_suggests_alternate_strategies(mocker, strategy):
    mocker.patch("src.summarization.stuff_strategy.get_llm")
    oversized_text = "x" * ((STUFF_STRATEGY_MAX_ESTIMATED_TOKENS + 10_000) * 4)
    chunks = [_make_chunk(oversized_text)]

    with pytest.raises(SummarizationError, match="Map-Reduce or Refine"):
        strategy.summarize(chunks)

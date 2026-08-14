"""
tests/test_refine_strategy.py

Tests for src/summarization/refine_strategy.py.

The key test in this file (test_refine_chain_sequentially_builds_on_previous_output)
proves TRUE sequential chaining, not just "N calls happened": the fake
refine LLM returns a NESTED string combining existing_summary and
new_content (e.g. refining "INITIAL" with "chunk2" produces
"[INITIAL]+[chunk2]"). After several refine calls, the final output's
nesting structure can only be correct if each call genuinely received
the PREVIOUS call's real output as its existing_summary -- a bug that
reused a stale or fixed summary would produce a visibly different,
wrong nesting.

get_llm() is called via side_effect=[...] lists sized EXACTLY to the
expected number of calls -- if the code called get_llm() more times
than expected (e.g. once per chunk instead of once total for the
refine chain), the mock would raise StopIteration, which is itself
a useful implicit assertion.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.summarization.base_strategy import SummarizationError, SummaryResult
from src.summarization.refine_strategy import RefineStrategy


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


def _fake_initial_llm(fixed_text="INITIAL"):
    """Stands in for ChatGroq during the initial (first-chunk) step."""
    def _run(prompt_value):
        return AIMessage(content=fixed_text)

    return RunnableLambda(_run)


def _fake_refine_llm(captured_calls=None):
    """
    Stands in for ChatGroq during the refine/update step. Parses
    EXISTING SUMMARY and NEW CONTENT out of the human message (per
    REFINE_UPDATE_PROMPT's known template) and returns a NEW summary
    that nests both together. This lets tests verify true sequential
    chaining by inspecting the final result's nesting structure.
    """
    def _run(prompt_value):
        human_content = prompt_value.to_messages()[-1].content
        if captured_calls is not None:
            captured_calls.append(human_content)
        existing = human_content.split("EXISTING SUMMARY:\n")[1].split(
            "\n\nNEW CONTENT:\n"
        )[0]
        new_content = human_content.split("\n\nNEW CONTENT:\n")[1]
        return AIMessage(content=f"[{existing}]+[{new_content}]")

    return RunnableLambda(_run)


@pytest.fixture
def strategy():
    return RefineStrategy()


# ---------------------------------------------------------------------------
# Single-chunk case -- no refine step should run at all
# ---------------------------------------------------------------------------

def test_single_chunk_uses_only_the_initial_call(mocker, strategy):
    # side_effect has exactly ONE item -- if refine_strategy.py called
    # get_llm() a second time for a single chunk, this mock would
    # raise StopIteration, failing the test.
    mocker.patch(
        "src.summarization.refine_strategy.get_llm",
        side_effect=[_fake_initial_llm("Only summary.")],
    )
    chunks = [_make_chunk("Only chunk.")]

    result = strategy.summarize(chunks)

    assert result.content == "Only summary."
    assert result.chunk_count == 1
    assert result.strategy == "refine"


# ---------------------------------------------------------------------------
# Multi-chunk case -- proves TRUE sequential dependency
# ---------------------------------------------------------------------------

def test_refine_chain_sequentially_builds_on_previous_output(mocker, strategy):
    captured = []
    mocker.patch(
        "src.summarization.refine_strategy.get_llm",
        side_effect=[_fake_initial_llm("INITIAL"), _fake_refine_llm(captured_calls=captured)],
    )
    chunks = [
        _make_chunk("chunk2 text", chunk_index=1),
        _make_chunk("chunk3 text", chunk_index=2),
    ]
    # NOTE: documents[0] below stands in as "chunk1" (the first
    # document passed IS the initial chunk); these two are the
    # remaining chunks refined in sequence.
    all_chunks = [_make_chunk("chunk1 text", chunk_index=0)] + chunks

    result = strategy.summarize(all_chunks)

    # The nesting can only be correct end-to-end if refine call #2
    # genuinely received refine call #1's real output as its
    # existing_summary -- not the original chunk1 text, not a fixed
    # placeholder, and not chunk3 processed before chunk2.
    assert result.content == "[[INITIAL]+[chunk2 text]]+[chunk3 text]"
    assert result.chunk_count == 3

    assert "INITIAL" in captured[0] and "chunk2 text" in captured[0]
    assert "[INITIAL]+[chunk2 text]" in captured[1] and "chunk3 text" in captured[1]


def test_refine_processes_chunks_in_original_order(mocker, strategy):
    captured = []
    mocker.patch(
        "src.summarization.refine_strategy.get_llm",
        side_effect=[_fake_initial_llm("start"), _fake_refine_llm(captured_calls=captured)],
    )
    chunks = [
        _make_chunk("first", chunk_index=0),
        _make_chunk("second", chunk_index=1),
        _make_chunk("third", chunk_index=2),
        _make_chunk("fourth", chunk_index=3),
    ]

    strategy.summarize(chunks)

    # Each captured call's NEW CONTENT should appear in the same order
    # as the input chunks: second, third, fourth.
    new_contents_in_call_order = [call.split("NEW CONTENT:\n")[1] for call in captured]
    assert new_contents_in_call_order == ["second", "third", "fourth"]


# ---------------------------------------------------------------------------
# Chain reuse -- proves the refine chain is built ONCE, not per chunk
# ---------------------------------------------------------------------------

def test_get_llm_called_exactly_twice_regardless_of_chunk_count(mocker, strategy):
    mock_get_llm = mocker.patch(
        "src.summarization.refine_strategy.get_llm",
        side_effect=[_fake_initial_llm("start"), _fake_refine_llm()],
    )
    chunks = [_make_chunk(f"chunk {i}", chunk_index=i) for i in range(5)]

    strategy.summarize(chunks)

    # Exactly 2 calls -- one for initial_chain, one for refine_chain --
    # NOT 5 (one per chunk). Proves refine_chain is built once outside
    # the loop and reused for every remaining chunk.
    assert mock_get_llm.call_count == 2


# ---------------------------------------------------------------------------
# Validation / metadata
# ---------------------------------------------------------------------------

def test_summarize_raises_on_empty_document_list(strategy):
    with pytest.raises(SummarizationError, match="no content"):
        strategy.summarize([])


def test_source_metadata_extracted_correctly(mocker, strategy):
    mocker.patch(
        "src.summarization.refine_strategy.get_llm",
        side_effect=[_fake_initial_llm("summary")],
    )
    chunks = [_make_chunk("Only chunk.", source_url="https://example.com/article")]

    result = strategy.summarize(chunks)

    assert result.source_metadata["source_type"] == "website"
    assert result.source_metadata["source_url"] == "https://example.com/article"
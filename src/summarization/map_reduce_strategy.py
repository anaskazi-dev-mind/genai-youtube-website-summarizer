"""
src/summarization/map_reduce_strategy.py

Map-Reduce strategy:
  1. Map step -- every chunk is summarized independently, CONCURRENTLY
     (via safe_batch / Runnable.batch()), producing one short
     intermediate summary per chunk.
  2. Reduce step -- all intermediate summaries are combined into one
     final structured summary with a single additional Groq call.

Built for content too long for the Stuff strategy's single call: N
smaller, concurrent calls plus one combining call, instead of one huge
call. See prompts.py for why the map step's output is deliberately
plain text, not the final structured format.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.config import MAP_REDUCE_MAX_CONCURRENCY, STUFF_STRATEGY_MAX_ESTIMATED_TOKENS
from src.logger import get_logger, truncate_for_log
from src.processing.chunker import estimate_token_count
from src.summarization.base_strategy import (
    SummarizationError,
    SummarizationStrategy,
    SummaryResult,
)
from src.summarization.llm import get_llm, safe_batch, safe_invoke
from src.summarization.prompts import MAP_PROMPT, REDUCE_PROMPT

logger = get_logger(__name__)


class MapReduceStrategy(SummarizationStrategy):
    """Map step (concurrent, per-chunk) followed by a single reduce/combine call."""

    name = "map_reduce"

    def summarize(self, documents: List[Document]) -> SummaryResult:
        self._validate_documents(documents)

        chunk_summaries = self._map_chunks(documents)
        final_summary = self._reduce_summaries(chunk_summaries)

        logger.info(
            "Map-Reduce strategy completed: %s", truncate_for_log(final_summary)
        )

        return SummaryResult(
            content=final_summary,
            strategy=self.name,
            chunk_count=len(documents),
            source_metadata=self._source_metadata_from(documents),
        )

    def _map_chunks(self, documents: List[Document]) -> List[str]:
        """
        Runs MAP_PROMPT against every chunk concurrently. Output order
        matches input order (verified directly against
        Runnable.batch()) -- chunk N's summary is always at index N,
        regardless of which call actually finished first.
        """
        logger.info(
            "Map-Reduce: mapping %d chunk(s) with max_concurrency=%d",
            len(documents),
            MAP_REDUCE_MAX_CONCURRENCY,
        )

        map_chain = MAP_PROMPT | get_llm() | StrOutputParser()
        map_inputs = [{"chunk_content": doc.page_content} for doc in documents]

        return safe_batch(
            map_chain, map_inputs, max_concurrency=MAP_REDUCE_MAX_CONCURRENCY
        )

    def _reduce_summaries(self, chunk_summaries: List[str]) -> str:
        """
        Combines all per-chunk summaries into one final structured
        summary with a single Groq call. Sections are numbered so the
        model can reference order if useful.

        Includes the same proactive size check as the Stuff strategy:
        an honest edge case is that, for content with an extremely
        large number of chunks, even the CONDENSED intermediate
        summaries combined could exceed a safe single-call size. This
        version does not implement a recursive/hierarchical reduce to
        handle that case -- it fails clearly instead of silently
        risking a Groq context-length error.
        """
        logger.info(
            "Map-Reduce: reducing %d chunk summaries into one", len(chunk_summaries)
        )

        numbered_summaries = "\n\n".join(
            f"[Section {i + 1}]\n{summary}" for i, summary in enumerate(chunk_summaries)
        )

        estimated_tokens = estimate_token_count(numbered_summaries)
        if estimated_tokens > STUFF_STRATEGY_MAX_ESTIMATED_TOKENS:
            raise SummarizationError(
                "Even after per-chunk summarization, the combined "
                f"intermediate summaries are too large for a single "
                f"reduce call (~{estimated_tokens:,} estimated tokens). "
                "This is an edge case for extremely long content with "
                "very many chunks -- a hierarchical reduce step would "
                "be needed to handle it, which isn't implemented in "
                "this version."
            )

        reduce_chain = REDUCE_PROMPT | get_llm() | StrOutputParser()
        return safe_invoke(reduce_chain, {"combined_summaries": numbered_summaries})

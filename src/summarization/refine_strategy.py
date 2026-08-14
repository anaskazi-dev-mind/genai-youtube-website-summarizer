"""
src/summarization/refine_strategy.py

Refine strategy: builds an initial summary from the first chunk, then
sequentially updates ("refines") it with each remaining chunk, one
Groq call at a time.

Fundamentally different from Map-Reduce's map step: this CANNOT be
parallelized. Each call depends on the PREVIOUS call's output (the
current running summary), so chunk 3 can only be processed after
chunk 2's refinement is done, which can only happen after chunk 1's
initial summary exists. This is a genuine, structural trade-off, not
a missed optimization -- see the WHY notes below and the file
walkthrough for what this costs in latency versus Map-Reduce.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.logger import get_logger, truncate_for_log
from src.summarization.base_strategy import SummarizationStrategy, SummaryResult
from src.summarization.llm import get_llm, safe_invoke
from src.summarization.prompts import REFINE_INITIAL_PROMPT, REFINE_UPDATE_PROMPT

logger = get_logger(__name__)


class RefineStrategy(SummarizationStrategy):
    """
    First chunk -> initial structured summary. Every remaining chunk,
    processed in order -> the existing summary is updated to fold in
    genuinely new information, one chunk at a time.
    """

    name = "refine"

    def summarize(self, documents: List[Document]) -> SummaryResult:
        self._validate_documents(documents)

        total_chunks = len(documents)
        logger.info("Refine: building initial summary from chunk 1 of %d", total_chunks)

        initial_chain = REFINE_INITIAL_PROMPT | get_llm() | StrOutputParser()
        current_summary = safe_invoke(
            initial_chain, {"first_chunk_content": documents[0].page_content}
        )

        if total_chunks > 1:
            # Built once, reused for every remaining chunk -- the chain
            # itself (prompt + llm + parser) is identical each time,
            # only the input changes. The loop below is deliberately
            # sequential (a plain for-loop, not safe_batch): each call
            # needs current_summary from the PREVIOUS iteration, so
            # there is nothing to parallelize here.
            refine_chain = REFINE_UPDATE_PROMPT | get_llm() | StrOutputParser()

            for index, chunk in enumerate(documents[1:], start=2):
                logger.info(
                    "Refine: updating summary with chunk %d of %d", index, total_chunks
                )
                current_summary = safe_invoke(
                    refine_chain,
                    {
                        "existing_summary": current_summary,
                        "new_content": chunk.page_content,
                    },
                )

        logger.info("Refine strategy completed: %s", truncate_for_log(current_summary))

        return SummaryResult(
            content=current_summary,
            strategy=self.name,
            chunk_count=total_chunks,
            source_metadata=self._source_metadata_from(documents),
        )
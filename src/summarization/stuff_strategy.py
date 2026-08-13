"""
src/summarization/stuff_strategy.py

Stuff strategy: combines every provided chunk's text into a single
block and summarizes it in exactly ONE Groq call.

Simplest of the three strategies -- one LLM call, no intermediate
combination step -- so it's fast and cheap for content that comfortably
fits in the model's context window. It's also the strategy most likely
to fail outright on very large content, since it has no fallback if
the combined text is too big. This file proactively checks estimated
size BEFORE calling Groq and fails with a clear, actionable message,
rather than sending an oversized request and letting Groq's API return
a cryptic context-length error.

Verified directly (see conversation): ChatPromptTemplate | llm |
StrOutputParser() composes correctly via LangChain's LCEL pipe
operator, and StrOutputParser's output behaves as a plain str.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.config import STUFF_STRATEGY_MAX_ESTIMATED_TOKENS
from src.logger import get_logger, truncate_for_log
from src.processing.chunker import estimate_token_count
from src.summarization.base_strategy import (
    SummarizationError,
    SummarizationStrategy,
    SummaryResult,
)
from src.summarization.llm import get_llm, safe_invoke
from src.summarization.prompts import STUFF_PROMPT

logger = get_logger(__name__)


class StuffStrategy(SummarizationStrategy):
    """
    Combines every provided chunk's text into one block and summarizes
    it in a single Groq call using STUFF_PROMPT.
    """

    name = "stuff"

    def summarize(self, documents: List[Document]) -> SummaryResult:
        self._validate_documents(documents)

        combined_content = "\n\n".join(doc.page_content for doc in documents)
        estimated_tokens = estimate_token_count(combined_content)

        if estimated_tokens > STUFF_STRATEGY_MAX_ESTIMATED_TOKENS:
            raise SummarizationError(
                f"This content is too long for the Stuff strategy "
                f"(estimated ~{estimated_tokens:,} tokens, limit is "
                f"{STUFF_STRATEGY_MAX_ESTIMATED_TOKENS:,} tokens). "
                "Please use Map-Reduce or Refine instead -- both are "
                "designed to handle long content in smaller pieces."
            )

        logger.info(
            "Stuff strategy: summarizing %d chunk(s), ~%d estimated "
            "tokens, in a single Groq call",
            len(documents),
            estimated_tokens,
        )

        chain = STUFF_PROMPT | get_llm() | StrOutputParser()
        summary_text = safe_invoke(chain, {"content": combined_content})

        logger.info("Stuff strategy completed: %s", truncate_for_log(summary_text))

        return SummaryResult(
            content=summary_text,
            strategy=self.name,
            chunk_count=len(documents),
            source_metadata=self._source_metadata_from(documents),
        )

"""
src/summarization/base_strategy.py

Defines the common interface every summarization strategy (Stuff,
Map-Reduce, Refine) implements, plus the shared SummaryResult shape
all three return. This is the seam that makes strategies swappable:
strategy_factory.py and app.py only ever depend on this interface,
never on a specific strategy's internals -- adding a fourth strategy
later means writing one new class here, not touching anything else.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document

# Metadata fields worth surfacing to the UI about WHERE the content
# came from. A fixed allowlist, not "copy everything" -- chunk-level
# fields like chunk_index/start_index are meaningless once
# summarization has combined all chunks into one result.
_DISPLAYABLE_METADATA_KEYS = {
    "source_type",
    "source_url",
    "video_id",
    "title",
    "language",
    "is_generated",
}


class SummarizationError(RuntimeError):
    """
    Raised for summarization pipeline errors that are NOT a Groq API
    failure (those are LLMGenerationError, from llm.py) -- e.g. being
    asked to summarize an empty document list. Carries a user-safe
    message separately from any underlying cause, matching the pattern
    used by every extraction error type in this project.
    """

    def __init__(self, user_message: str, *, cause: Optional[Exception] = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.cause = cause


@dataclass(frozen=True)
class SummaryResult:
    """
    The output every strategy returns. Deliberately does NOT parse the
    Markdown summary into individual Python fields (title, key_points,
    etc.) -- the LLM already produces well-formed Markdown per the
    structure defined in prompts.py, and Streamlit renders Markdown
    natively via st.markdown(). Parsing it into a rigid schema would
    add a fragile layer that could break on minor formatting
    variation, for no real benefit over rendering the Markdown as-is.

    Note: frozen=True prevents reassigning these attributes, but
    source_metadata is still a plain dict, so its CONTENTS remain
    mutable -- frozen dataclasses give shallow immutability, not deep.
    """

    content: str  # the structured Markdown summary text
    strategy: str  # "stuff" | "map_reduce" | "refine"
    chunk_count: int  # how many chunks were actually summarized
    source_metadata: dict  # source_type, source_url, title/video_id, etc.


class SummarizationStrategy(ABC):
    """
    Common interface every summarization strategy implements. Callers
    (strategy_factory.py, and ultimately app.py) depend only on this
    interface, never on Stuff/Map-Reduce/Refine's internals.
    """

    #: Short machine-readable name, e.g. "stuff". Each subclass sets
    #: this; used in SummaryResult.strategy and in logs.
    name: str

    @abstractmethod
    def summarize(self, documents: List[Document]) -> SummaryResult:
        """
        Given a list of already-cleaned, already-chunked (and, where
        applicable, already-deduplicated) Documents, returns a
        SummaryResult.

        Implementations decide internally how many LLM calls to make
        and how to combine partial results -- that decision is the
        entire difference between Stuff, Map-Reduce, and Refine, and
        it is fully encapsulated behind this one method.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_documents(documents: List[Document]) -> None:
        """Raises SummarizationError if there's nothing to summarize."""
        if not documents:
            raise SummarizationError(
                "There is no content to summarize. This usually means "
                "extraction or chunking produced no usable text."
            )

    @staticmethod
    def _source_metadata_from(documents: List[Document]) -> dict:
        """
        Extracts display-relevant, source-level metadata (source_type,
        source_url, title/video_id, etc.) from the first document. All
        chunks from a single extraction share the same source-level
        metadata -- only chunk_index/start_index differ per chunk --
        so the first document is representative of the whole batch.
        """
        if not documents:
            return {}
        first_metadata = documents[0].metadata
        return {
            key: value
            for key, value in first_metadata.items()
            if key in _DISPLAYABLE_METADATA_KEYS
        }

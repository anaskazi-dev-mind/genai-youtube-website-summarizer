"""
src/processing/chunker.py

Splits cleaned Document objects into smaller chunks. Chunking is
needed for two reasons: (1) Map-Reduce and Refine are strategies that
operate per-chunk by design, regardless of whether the full content
would fit in a single LLM call, and (2) it's a safety net against any
single input exceeding Groq's context window, even for Stuff.

Uses LangChain's RecursiveCharacterTextSplitter, which tries natural
break points (paragraph, then sentence, then word) before falling back
to a hard character cut -- so chunks avoid splitting mid-sentence
whenever the content allows it.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.logger import get_logger

logger = get_logger(__name__)

# Character-based chunking, not token-based. We estimate ~4 characters
# per token for English text -- a standard, widely-used approximation.
# We deliberately don't count exact tokens against Groq's gpt-oss-120b
# tokenizer here: that tokenizer isn't published as a standard tiktoken
# encoding we could measure against precisely, so pretending to exact
# token counts would be false precision, not real accuracy.
_ESTIMATED_CHARS_PER_TOKEN = 4


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits one or more Documents into chunks sized by CHUNK_SIZE /
    CHUNK_OVERLAP (src/config.py). Each output chunk keeps the source
    Document's original metadata, plus:
      - start_index: character offset within the source text
      - chunk_index: this chunk's position among chunks from the same
        source_url (0-based)

    so any chunk can always be traced back to where it came from --
    useful in logs, and for the Refine strategy, which needs to know
    chunk order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # chunk_index is scoped per source_url, not global across all
    # input documents -- matters if this is ever called with multiple
    # documents from different sources at once.
    counters = {}
    for chunk in chunks:
        source_key = chunk.metadata.get("source_url", "unknown")
        chunk_index = counters.get(source_key, 0)
        chunk.metadata["chunk_index"] = chunk_index
        counters[source_key] = chunk_index + 1

    total_chars = sum(len(doc.page_content) for doc in documents)
    logger.info(
        "Split %d document(s) (%d chars, ~%d estimated tokens) into "
        "%d chunk(s) (chunk_size=%d, chunk_overlap=%d)",
        len(documents),
        total_chars,
        total_chars // _ESTIMATED_CHARS_PER_TOKEN,
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    return chunks


def estimate_token_count(text: str) -> int:
    """
    Rough token count estimate using the ~4-chars-per-token heuristic.
    Used for logging, and by the Stuff strategy (next milestone) to
    decide whether content is small enough for a single LLM call.
    Explicitly an ESTIMATE, never presented as a measured value.
    """
    return len(text) // _ESTIMATED_CHARS_PER_TOKEN

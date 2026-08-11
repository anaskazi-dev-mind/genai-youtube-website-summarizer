"""
src/processing/cleaner.py

Text cleaning applied to extracted content (YouTube transcripts or
website text) before chunking. Deliberately conservative: this module
removes obvious noise (bracketed non-speech annotations, excess
whitespace, encoding artifacts) without doing anything that could
change the actual meaning of the content, like sentence restructuring
or stopword removal.
"""

import re
import unicodedata

from langchain_core.documents import Document

from src.logger import get_logger, truncate_for_log

logger = get_logger(__name__)

# Bracketed non-speech annotations that sometimes appear in YouTube
# transcripts: [Music], [Applause], [Laughter], [inaudible], [crosstalk].
# These carry no summarizable meaning and are safe to remove outright.
_NON_SPEECH_ANNOTATION_PATTERN = re.compile(
    r"\[(music|applause|laughter|inaudible|crosstalk|silence)\]",
    re.IGNORECASE,
)

# Two or more consecutive whitespace characters (spaces, tabs, newlines)
# collapse to a single space. We deliberately don't preserve paragraph
# breaks here -- the text splitter downstream chunks by character
# count, not paragraph structure, so paragraph boundaries aren't
# load-bearing for this pipeline.
_EXCESS_WHITESPACE_PATTERN = re.compile(r"\s{2,}")


def clean_text(text: str | None) -> str:
    """
    Applies conservative, source-agnostic cleaning:
      1. Unicode normalization (NFKC) -- fixes encoding artifacts like
         curly quotes represented as separate combining characters, and
         collapses visually-identical character variants to one form.
      2. Non-breaking spaces (U+00A0), common in scraped HTML, replaced
         with regular spaces.
      3. Bracketed non-speech annotations removed.
      4. Excess whitespace collapsed to single spaces.
      5. Leading/trailing whitespace stripped.

    Returns an empty string unchanged for empty input -- this function
    does not raise. Both extractors already reject empty content before
    this point, so this is a defensive default, not the primary
    empty-content check.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = _NON_SPEECH_ANNOTATION_PATTERN.sub(" ", text)
    text = _EXCESS_WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_document(document: Document) -> Document:
    """
    Returns a NEW Document with cleaned page_content and unchanged
    metadata. Returns a new object rather than mutating in place, so
    the original extracted Document is still available if ever needed
    (e.g. comparing raw vs. cleaned length for debugging).
    """
    cleaned_content = clean_text(document.page_content)

    logger.info(
        "Cleaned document (source=%s): %d chars -> %d chars: %s",
        document.metadata.get("source_type", "unknown"),
        len(document.page_content),
        len(cleaned_content),
        truncate_for_log(cleaned_content),
    )

    return Document(page_content=cleaned_content, metadata=dict(document.metadata))

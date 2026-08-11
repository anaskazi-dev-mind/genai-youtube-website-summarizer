"""
tests/test_cleaner.py

Tests for src/processing/cleaner.py.

Pure-function tests -- no mocking needed. Each test proves one specific
cleaning claim: whitespace collapsing, non-breaking space replacement,
non-speech annotation removal (and that it's a whitelist, not a
catch-all), Unicode normalization, and Document immutability.
"""

from langchain_core.documents import Document

from src.processing.cleaner import clean_document, clean_text

# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


def test_clean_text_collapses_excess_whitespace():
    assert (
        clean_text("This   has\n\nexcess    whitespace.")
        == "This has excess whitespace."
    )


def test_clean_text_strips_leading_and_trailing_whitespace():
    assert clean_text("   padded text   ") == "padded text"


def test_clean_text_replaces_non_breaking_spaces():
    text_with_nbsp = "This\xa0has\xa0non-breaking\xa0spaces."
    assert clean_text(text_with_nbsp) == "This has non-breaking spaces."


def test_clean_text_removes_known_non_speech_annotations():
    transcript = "Welcome to the show. [Music] Today we discuss AI. [Applause]"
    assert clean_text(transcript) == "Welcome to the show. Today we discuss AI."


def test_clean_text_annotation_removal_is_case_insensitive():
    assert clean_text("Hello [MUSIC] world") == "Hello world"


def test_clean_text_does_not_remove_unrelated_bracketed_content():
    # Proves the annotation removal is a specific whitelist, not a
    # blanket "delete anything in brackets" rule -- content like [sic]
    # or a bracketed reference number is real content, not noise.
    text = "The report claims 40% growth [sic] and cites [42] as evidence."
    cleaned = clean_text(text)
    assert "[sic]" in cleaned
    assert "[42]" in cleaned


def test_clean_text_normalizes_unicode_to_canonical_form():
    # "e" + combining acute accent (U+0301) is visually identical to
    # the single precomposed "é" (U+00E9) character, but is a
    # different underlying sequence. NFKC normalization should
    # compose it into the single canonical character.
    decomposed = "cafe\u0301"  # é represented as two code points
    precomposed = "café"  # é as a single code point
    assert clean_text(decomposed) == precomposed


def test_clean_text_handles_empty_and_none_input_safely():
    assert clean_text("") == ""
    assert clean_text(None) == ""


# ---------------------------------------------------------------------------
# clean_document
# ---------------------------------------------------------------------------


def test_clean_document_cleans_content_and_preserves_metadata():
    original = Document(
        page_content="Messy   text  with\xa0noise. [Music]",
        metadata={"source_type": "youtube", "video_id": "abc123"},
    )

    cleaned = clean_document(original)

    assert cleaned.page_content == "Messy text with noise."
    assert cleaned.metadata == original.metadata


def test_clean_document_does_not_mutate_the_original():
    original = Document(
        page_content="Messy   text.",
        metadata={"source_type": "website"},
    )
    original_content_before = original.page_content

    clean_document(original)

    assert original.page_content == original_content_before


def test_clean_document_returns_a_different_object():
    original = Document(page_content="Some text.", metadata={"source_type": "website"})
    cleaned = clean_document(original)
    assert cleaned is not original

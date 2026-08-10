"""
tests/test_youtube_extractor.py

Tests for src/extractors/youtube_extractor.py.

YouTubeTranscriptApi.fetch() is mocked in every test -- no real network
calls are made. Each test proves one specific claim: either a specific
library exception maps to the correct user-facing YouTubeExtractionError
message, or a successful fetch produces a correctly-populated Document.

Fake transcript data is built using the library's own FetchedTranscript /
FetchedTranscriptSnippet dataclasses (not a hand-rolled stand-in), so
these tests would fail loudly if a future version of the library changes
that data shape -- a hand-rolled stub could silently drift out of sync
and keep "passing" against a shape the real library no longer produces.
"""

import pytest
from youtube_transcript_api import (
    AgeRestricted,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    PoTokenRequired,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnplayable,
)
from youtube_transcript_api._transcripts import (
    FetchedTranscript,
    FetchedTranscriptSnippet,
)

from src.extractors.youtube_extractor import (
    YouTubeExtractionError,
    fetch_youtube_document,
)
from src.validators import URLValidationError

_TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_TEST_VIDEO_ID = "dQw4w9WgXcQ"


def _make_fetched_transcript(text_parts, language_code="en", is_generated=True):
    """Builds a real FetchedTranscript using the library's own dataclasses."""
    snippets = [
        FetchedTranscriptSnippet(text=text, start=i * 2.0, duration=2.0)
        for i, text in enumerate(text_parts)
    ]
    return FetchedTranscript(
        snippets=snippets,
        video_id=_TEST_VIDEO_ID,
        language="English",
        language_code=language_code,
        is_generated=is_generated,
    )


@pytest.fixture
def mock_transcript_api(mocker):
    """
    Patches the YouTubeTranscriptApi class where it's imported (inside
    youtube_extractor.py), and returns the mock instance's .fetch
    method so each test can configure return_value or side_effect.
    """
    mock_class = mocker.patch("src.extractors.youtube_extractor.YouTubeTranscriptApi")
    return mock_class.return_value.fetch


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_fetch_youtube_document_returns_populated_document(mock_transcript_api):
    mock_transcript_api.return_value = _make_fetched_transcript(
        ["Hello there.", "This is a test video."],
        language_code="en",
        is_generated=True,
    )

    document = fetch_youtube_document(_TEST_URL)

    assert document.page_content == "Hello there. This is a test video."
    assert document.metadata["source_type"] == "youtube"
    assert document.metadata["video_id"] == _TEST_VIDEO_ID
    assert document.metadata["source_url"] == _TEST_URL
    assert document.metadata["language"] == "en"
    assert document.metadata["is_generated"] is True


def test_fetch_youtube_document_requests_english_language_preference(
    mock_transcript_api,
):
    mock_transcript_api.return_value = _make_fetched_transcript(["Some text."])

    fetch_youtube_document(_TEST_URL)

    (called_video_id,) = mock_transcript_api.call_args.args
    assert called_video_id == _TEST_VIDEO_ID
    assert "en" in mock_transcript_api.call_args.kwargs["languages"]


# ---------------------------------------------------------------------------
# Named failure modes -> specific YouTubeExtractionError messages
# ---------------------------------------------------------------------------


def test_transcripts_disabled_raises_clear_message(mock_transcript_api):
    mock_transcript_api.side_effect = TranscriptsDisabled(_TEST_VIDEO_ID)

    with pytest.raises(YouTubeExtractionError, match="disabled"):
        fetch_youtube_document(_TEST_URL)


def test_no_transcript_found_raises_clear_message(mock_transcript_api, mocker):
    mock_transcript_api.side_effect = NoTranscriptFound(
        _TEST_VIDEO_ID, ["en"], mocker.MagicMock()
    )

    with pytest.raises(YouTubeExtractionError, match="No transcript"):
        fetch_youtube_document(_TEST_URL)


def test_age_restricted_raises_clear_message(mock_transcript_api):
    mock_transcript_api.side_effect = AgeRestricted(_TEST_VIDEO_ID)

    with pytest.raises(YouTubeExtractionError, match="age-restricted"):
        fetch_youtube_document(_TEST_URL)


@pytest.mark.parametrize(
    "exception",
    [
        VideoUnplayable(_TEST_VIDEO_ID, reason=None, sub_reasons=[]),
        InvalidVideoId(_TEST_VIDEO_ID),
    ],
)
def test_unplayable_or_invalid_video_raises_clear_message(
    mock_transcript_api, exception
):
    mock_transcript_api.side_effect = exception

    with pytest.raises(YouTubeExtractionError, match="unavailable"):
        fetch_youtube_document(_TEST_URL)


@pytest.mark.parametrize(
    "exception",
    [RequestBlocked(_TEST_VIDEO_ID), IpBlocked(_TEST_VIDEO_ID)],
)
def test_blocked_requests_raise_clear_message(mock_transcript_api, exception):
    mock_transcript_api.side_effect = exception

    with pytest.raises(YouTubeExtractionError, match="blocking"):
        fetch_youtube_document(_TEST_URL)


def test_unnamed_library_exception_is_caught_by_base_class(mock_transcript_api):
    # PoTokenRequired is deliberately NOT special-cased in
    # youtube_extractor.py -- this proves the CouldNotRetrieveTranscript
    # catch-all still converts it to our exception type instead of
    # letting the raw library exception escape to the UI layer.
    mock_transcript_api.side_effect = PoTokenRequired(_TEST_VIDEO_ID)

    with pytest.raises(YouTubeExtractionError, match="could not be retrieved"):
        fetch_youtube_document(_TEST_URL)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_transcript_raises_clear_message(mock_transcript_api):
    mock_transcript_api.return_value = _make_fetched_transcript(["", "   "])

    with pytest.raises(YouTubeExtractionError, match="empty"):
        fetch_youtube_document(_TEST_URL)


def test_invalid_url_raises_url_validation_error_not_extraction_error(
    mock_transcript_api,
):
    # A URL with no video ID should fail in validators.py, before the
    # (mocked) network call ever happens -- proving the two error
    # types stay distinct at this boundary.
    with pytest.raises(URLValidationError):
        fetch_youtube_document("https://www.youtube.com/")

    mock_transcript_api.assert_not_called()

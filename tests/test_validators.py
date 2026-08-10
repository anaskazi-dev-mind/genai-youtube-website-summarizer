"""
tests/test_validators.py

Tests for src/validators.py.

These are pure-function tests -- no mocking needed, since validators.py
makes no network calls. Each test proves one specific claim:

- is_valid_url: structural URL validity, independent of reachability
- detect_source_type: correct YouTube-vs-website classification
- extract_youtube_video_id: correct ID extraction across every YouTube
  URL shape we claim to support, and a clear failure when none is found
"""

import pytest

from src.validators import (
    SourceType,
    URLValidationError,
    detect_source_type,
    extract_youtube_video_id,
    is_valid_url,
)

# ---------------------------------------------------------------------------
# is_valid_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "http://example.com",
        "https://example.com/some/path?query=1",
        "https://youtu.be/dQw4w9WgXcQ",
    ],
)
def test_is_valid_url_accepts_well_formed_http_urls(raw_url):
    assert is_valid_url(raw_url) is True


@pytest.mark.parametrize(
    "raw_url",
    [
        "",
        "   ",
        "not a url at all",
        "ftp://example.com",  # wrong scheme
        "www.example.com",  # missing scheme entirely
        "javascript:alert('x')",  # wrong scheme, and unsafe
        None,
    ],
)
def test_is_valid_url_rejects_malformed_or_unsupported_input(raw_url):
    # is_valid_url must never raise -- it's meant to be safe to call
    # directly on raw, untrusted user input from the Streamlit text box.
    assert is_valid_url(raw_url) is False


def test_is_valid_url_strips_surrounding_whitespace():
    assert is_valid_url("   https://example.com   ") is True


# ---------------------------------------------------------------------------
# detect_source_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
    ],
)
def test_detect_source_type_identifies_youtube_hosts(raw_url):
    assert detect_source_type(raw_url) == SourceType.YOUTUBE


@pytest.mark.parametrize(
    "raw_url",
    [
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://www.bbc.com/news",
        "http://example.com/article",
    ],
)
def test_detect_source_type_identifies_generic_websites(raw_url):
    assert detect_source_type(raw_url) == SourceType.WEBSITE


def test_detect_source_type_raises_on_invalid_url():
    with pytest.raises(URLValidationError):
        detect_source_type("not a url")


# ---------------------------------------------------------------------------
# extract_youtube_video_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_url, expected_id",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=30", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_youtube_video_id_supports_known_url_shapes(raw_url, expected_id):
    assert extract_youtube_video_id(raw_url) == expected_id


@pytest.mark.parametrize(
    "raw_url",
    [
        "https://www.youtube.com/",  # homepage, no video
        "https://www.youtube.com/@SomeChannel",  # channel, no video
        "https://www.youtube.com/watch?list=PL123",  # playlist, no v=
        "https://www.youtube.com/watch?v=tooshort",  # malformed ID length
    ],
)
def test_extract_youtube_video_id_raises_when_no_id_found(raw_url):
    with pytest.raises(URLValidationError):
        extract_youtube_video_id(raw_url)

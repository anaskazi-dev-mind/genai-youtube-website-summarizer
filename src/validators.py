"""
src/validators.py

URL validation and source-type detection.

This is the first real decision point in the pipeline: given a raw
string from the Streamlit text input, is it a URL at all, and if so,
is it a YouTube video or a generic website? Everything downstream
(which extractor runs) depends on getting this right, so it's kept as
a small set of pure functions with no side effects -- easy to unit
test exhaustively without mocking anything.
"""

import re
from enum import Enum
from urllib.parse import urlparse, parse_qs

from src.logger import get_logger

logger = get_logger(__name__)


class SourceType(str, Enum):
    """The two content sources this app knows how to handle."""

    YOUTUBE = "youtube"
    WEBSITE = "website"


class URLValidationError(ValueError):
    """Raised when a URL fails validation or has an unsupported form."""


# Hostnames that count as "YouTube" for source detection. Includes the
# short-link domain and the mobile/music subdomains users commonly paste.
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}

# YouTube video IDs are always exactly 11 characters from this alphabet.
_VIDEO_ID_PATTERN = r"[A-Za-z0-9_-]{11}"

# Each extractor tries one known YouTube URL shape and returns a
# candidate ID (or None). Tried in order until one matches.
_YOUTUBE_ID_EXTRACTORS = (
    # https://www.youtube.com/watch?v=VIDEOID
    lambda parsed: parse_qs(parsed.query).get("v", [None])[0],
    # https://youtu.be/VIDEOID
    lambda parsed: (
        parsed.path.lstrip("/").split("/")[0]
        if parsed.netloc.endswith("youtu.be")
        else None
    ),
    # https://www.youtube.com/embed/VIDEOID, /shorts/VIDEOID, /live/VIDEOID
    lambda parsed: next(
        (
            parsed.path.split(f"/{segment}/")[1].split("/")[0]
            for segment in ("embed", "shorts", "live")
            if f"/{segment}/" in parsed.path
        ),
        None,
    ),
)


def is_valid_url(raw_url: str) -> bool:
    """
    Structural validation only: is this a well-formed http(s) URL?
    Does not check reachability -- that's the extractor's job, since
    "reachable" requires a network call and this function must stay
    fast and side-effect-free for use in the UI's input validation.
    """
    if not raw_url or not raw_url.strip():
        return False

    parsed = urlparse(raw_url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def detect_source_type(raw_url: str) -> SourceType:
    """
    Classifies a validated URL as YOUTUBE or WEBSITE based on hostname.
    Raises URLValidationError if the URL isn't structurally valid --
    callers should call is_valid_url() first if they want to show a
    specific "invalid URL" message in the UI rather than catch this.
    """
    if not is_valid_url(raw_url):
        raise URLValidationError(f"'{raw_url}' is not a valid http(s) URL.")

    hostname = (urlparse(raw_url.strip()).hostname or "").lower()
    if hostname in _YOUTUBE_HOSTS:
        return SourceType.YOUTUBE
    return SourceType.WEBSITE


def extract_youtube_video_id(raw_url: str) -> str:
    """
    Extracts the 11-character video ID from any supported YouTube URL
    shape (watch, short link, embed, shorts, live).

    Raises URLValidationError if the URL is YouTube but no video ID
    could be found (e.g. a channel URL or the bare youtube.com homepage)
    -- a deliberate, named failure rather than returning None, since
    the caller has no valid path forward without a video ID.
    """
    parsed = urlparse(raw_url.strip())

    candidate = None
    for extractor in _YOUTUBE_ID_EXTRACTORS:
        candidate = extractor(parsed)
        if candidate:
            break

    if not candidate or not re.fullmatch(_VIDEO_ID_PATTERN, candidate):
        raise URLValidationError(
            f"Could not find a video ID in '{raw_url}'. "
            "Supported formats: youtube.com/watch?v=..., youtu.be/..., "
            "youtube.com/shorts/..., youtube.com/embed/..."
        )

    return candidate

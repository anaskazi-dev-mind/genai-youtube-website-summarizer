"""
src/extractors/youtube_extractor.py

Fetches a transcript for a YouTube video and converts it into a
LangChain Document.

Uses the instance-based youtube-transcript-api 1.x interface
(YouTubeTranscriptApi().fetch(...)), not the older classmethod-based
API (YouTubeTranscriptApi.get_transcript(...)) that most tutorials
still show but no longer reflects how the installed 1.2.4 library
actually works -- verified directly against the installed package,
not assumed.
"""

from typing import Optional

from langchain_core.documents import Document
from youtube_transcript_api import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnplayable,
    YouTubeTranscriptApi,
)

from src.logger import get_logger, truncate_for_log
from src.validators import extract_youtube_video_id

logger = get_logger(__name__)

# English first, since it's the expected common case for this project's
# test videos. Listing variants lets youtube-transcript-api match
# whichever English track a video actually has, instead of failing on
# an exact "en" mismatch.
_PREFERRED_LANGUAGES = ("en", "en-US", "en-GB")


class YouTubeExtractionError(RuntimeError):
    """
    Raised for every YouTube extraction failure. Carries a short,
    user-safe message (shown directly in the Streamlit UI) separately
    from the original library exception (kept for logs) -- a user
    should never see a raw library stack trace, but the log line still
    has enough detail to debug.
    """

    def __init__(self, user_message: str, *, cause: Optional[Exception] = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.cause = cause


def fetch_youtube_document(url: str) -> Document:
    """
    Given a YouTube URL, returns a single LangChain Document containing
    the full transcript text, with metadata identifying the source.

    Raises YouTubeExtractionError for every failure mode we can name
    (disabled transcripts, no transcript in any supported language,
    blocked/rate-limited requests, invalid or unplayable video,
    age-restricted video, empty transcript) with a message safe to
    show directly in the UI.
    """
    video_id = extract_youtube_video_id(url)
    logger.info("Fetching YouTube transcript for video_id=%s", video_id)

    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=_PREFERRED_LANGUAGES)
    except TranscriptsDisabled as exc:
        raise YouTubeExtractionError(
            "This video's owner has disabled transcripts/captions, so "
            "it can't be summarized from its transcript.",
            cause=exc,
        ) from exc
    except NoTranscriptFound as exc:
        raise YouTubeExtractionError(
            "No transcript is available for this video in English. "
            "Non-English-only videos aren't supported yet.",
            cause=exc,
        ) from exc
    except AgeRestricted as exc:
        raise YouTubeExtractionError(
            "This video is age-restricted, and its transcript can't be "
            "retrieved without a signed-in session.",
            cause=exc,
        ) from exc
    except (VideoUnplayable, InvalidVideoId) as exc:
        raise YouTubeExtractionError(
            "This video is unavailable or the link doesn't point to a "
            "playable video. Double-check the URL and try again.",
            cause=exc,
        ) from exc
    except (RequestBlocked, IpBlocked) as exc:
        raise YouTubeExtractionError(
            "YouTube is currently blocking transcript requests from "
            "this server's network. This is a known limitation when "
            "running from cloud infrastructure -- see the README for "
            "details.",
            cause=exc,
        ) from exc
    except CouldNotRetrieveTranscript as exc:
        # Catch-all for any other named failure in the library's
        # exception hierarchy that we haven't special-cased above.
        raise YouTubeExtractionError(
            "The transcript for this video could not be retrieved.",
            cause=exc,
        ) from exc

    transcript_text = " ".join(snippet.text for snippet in fetched).strip()

    if not transcript_text:
        raise YouTubeExtractionError(
            "The transcript for this video was empty after fetching -- "
            "there may be no usable speech content."
        )

    logger.info(
        "Fetched transcript for video_id=%s (%d chars, language=%s, "
        "auto_generated=%s): %s",
        video_id,
        len(transcript_text),
        fetched.language_code,
        fetched.is_generated,
        truncate_for_log(transcript_text),
    )

    return Document(
        page_content=transcript_text,
        metadata={
            "source_type": "youtube",
            "video_id": video_id,
            "source_url": url,
            "language": fetched.language_code,
            "is_generated": fetched.is_generated,
        },
    )

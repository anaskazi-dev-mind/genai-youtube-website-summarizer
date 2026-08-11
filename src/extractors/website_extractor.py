"""
src/extractors/website_extractor.py

Fetches a webpage and extracts its main textual content (stripping
navigation, ads, and other boilerplate), returning a LangChain Document.

Primary extraction is trafilatura, which is purpose-built for
main-content detection. When trafilatura can't confidently extract
enough text (verified behavior, not assumed -- see file walkthrough),
a BeautifulSoup-based fallback pulls all <p> tag text after stripping
obviously non-content tags.
"""

import json

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.logger import get_logger, truncate_for_log
from src.validators import URLValidationError, is_valid_url
from src.config import MIN_ACCEPTABLE_TEXT_LENGTH

logger = get_logger(__name__)

_REQUEST_TIMEOUT_SECONDS = 15
_USER_AGENT = (
    "Mozilla/5.0 (compatible; AI-YouTube-Website-Summarizer/1.0; "
    "student portfolio project)"
)
# Below this many characters, we don't trust the extraction -- likely a
# JS-rendered shell, an error/consent page, or a page with genuinely
# little content. This is a practical heuristic threshold, not a value
# validated against a labeled dataset -- documented honestly as such.


class WebsiteExtractionError(RuntimeError):
    """
    Raised for every website extraction failure. Carries a short,
    user-safe message (shown in the Streamlit UI) separately from the
    original exception (kept for logs).
    """

    def __init__(self, user_message: str, *, cause: Exception | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.cause = cause


def _fetch_html(url: str) -> str:
    """Fetches raw HTML with a timeout and a real User-Agent, or raises
    WebsiteExtractionError with a message specific to the failure."""
    headers = {"User-Agent": _USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise WebsiteExtractionError(
            "The website took too long to respond and the request timed out.",
            cause=exc,
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise WebsiteExtractionError(
            "Could not connect to this website. It may be down, or it "
            "may be blocking automated requests.",
            cause=exc,
        ) from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise WebsiteExtractionError(
            f"The website returned an error (HTTP {status}) and the "
            "page could not be fetched.",
            cause=exc,
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise WebsiteExtractionError(
            "Something went wrong while fetching this website.",
            cause=exc,
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        raise WebsiteExtractionError(
            "This URL doesn't point to a webpage (content type: "
            f"'{content_type or 'unknown'}'). Only HTML pages are supported."
        )

    return response.text


def _extract_with_beautifulsoup(html: str) -> tuple[str, str | None]:
    """Fallback content extraction: strip obviously non-content tags,
    then join whatever text remains in <p> tags."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "form"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if p)
    return text.strip(), title


def _extract_main_content(html: str) -> tuple[str, str | None]:
    """Returns (text, title). Tries trafilatura first; falls back to
    BeautifulSoup if trafilatura returns nothing or too little text."""
    json_result = trafilatura.extract(
        html,
        output_format="json",
        with_metadata=True,
        include_comments=False,
        include_tables=True,
    )

    if json_result:
        try:
            parsed = json.loads(json_result)
            text = (parsed.get("text") or "").strip()
            title = parsed.get("title")
            if len(text) >= MIN_ACCEPTABLE_TEXT_LENGTH:
                return text, title
        except json.JSONDecodeError:
            pass  # fall through to the BeautifulSoup fallback below

    return _extract_with_beautifulsoup(html)


def fetch_website_document(url: str) -> Document:
    """
    Given a website URL, returns a single LangChain Document containing
    its main textual content, with metadata identifying the source.

    Raises URLValidationError if the URL is structurally invalid, or
    WebsiteExtractionError for every fetch/extraction failure mode
    (timeout, connection failure, non-HTML content, HTTP error,
    insufficient extractable text).
    """
    if not is_valid_url(url):
        raise URLValidationError(f"'{url}' is not a valid http(s) URL.")

    logger.info("Fetching website content for url=%s", url)

    html = _fetch_html(url)
    text, title = _extract_main_content(html)

    if not text or len(text) < MIN_ACCEPTABLE_TEXT_LENGTH:
        raise WebsiteExtractionError(
            "Couldn't extract meaningful article content from this "
            "page. It may have very little text, or it may rely "
            "heavily on JavaScript to render its content, which this "
            "app can't execute."
        )

    logger.info(
        "Extracted website content for url=%s (%d chars, title=%r): %s",
        url,
        len(text),
        title,
        truncate_for_log(text),
    )

    return Document(
        page_content=text,
        metadata={
            "source_type": "website",
            "source_url": url,
            "title": title or "Untitled",
        },
    )

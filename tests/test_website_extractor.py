"""
tests/test_website_extractor.py

Tests for src/extractors/website_extractor.py.

requests.get() is mocked in every test -- no real network calls are
made. trafilatura.extract() and BeautifulSoup are exercised for real
(not mocked) against small, real HTML strings, since they're pure
functions of their HTML input and testing them for real is both safe
and more meaningful than mocking a parser.
"""

import requests
import pytest

from src.extractors.website_extractor import (
    WebsiteExtractionError,
    fetch_website_document,
)
from src.validators import URLValidationError

_TEST_URL = "https://example.com/article"

_ARTICLE_HTML = """
<html>
<head><title>Deep Learning Explained</title></head>
<body>
<nav>Home | About | Contact</nav>
<article>
<h1>Deep Learning Explained</h1>
<p>Deep learning is a subset of machine learning based on artificial
neural networks with multiple layers, capable of learning complex
patterns directly from large amounts of raw data without extensive
manual feature engineering by practitioners.</p>
<p>These networks have driven major advances in computer vision,
natural language processing, and speech recognition over the last
decade, powering many of the AI systems in common use today.</p>
</article>
<footer>Copyright 2026 Example Corp. All rights reserved.</footer>
</body>
</html>
"""

_THIN_HTML = "<html><body><p>Hi.</p></body></html>"


def _make_mock_response(
    mocker, *, text="", status_code=200, content_type="text/html; charset=utf-8"
):
    response = mocker.MagicMock()
    response.text = text
    response.status_code = status_code
    response.headers = {"Content-Type": content_type}
    response.raise_for_status = mocker.MagicMock()
    return response


@pytest.fixture
def mock_requests_get(mocker):
    return mocker.patch("src.extractors.website_extractor.requests.get")


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_fetch_website_document_extracts_article_content(mock_requests_get, mocker):
    mock_requests_get.return_value = _make_mock_response(mocker, text=_ARTICLE_HTML)

    document = fetch_website_document(_TEST_URL)

    assert "Deep learning is a subset of machine learning" in document.page_content
    assert "Home | About | Contact" not in document.page_content
    assert "Copyright 2026 Example Corp" not in document.page_content
    assert document.metadata["source_type"] == "website"
    assert document.metadata["source_url"] == _TEST_URL
    assert document.metadata["title"] == "Deep Learning Explained"


def test_fetch_website_document_sends_a_real_user_agent(mock_requests_get, mocker):
    mock_requests_get.return_value = _make_mock_response(mocker, text=_ARTICLE_HTML)

    fetch_website_document(_TEST_URL)

    _, kwargs = mock_requests_get.call_args
    assert "User-Agent" in kwargs["headers"]
    assert kwargs["timeout"] > 0


# ---------------------------------------------------------------------------
# Network-level failures
# ---------------------------------------------------------------------------


def test_timeout_raises_clear_message(mock_requests_get):
    mock_requests_get.side_effect = requests.exceptions.Timeout()

    with pytest.raises(WebsiteExtractionError, match="timed out"):
        fetch_website_document(_TEST_URL)


def test_connection_error_raises_clear_message(mock_requests_get):
    mock_requests_get.side_effect = requests.exceptions.ConnectionError()

    with pytest.raises(WebsiteExtractionError, match="Could not connect"):
        fetch_website_document(_TEST_URL)


def test_http_error_includes_status_code(mock_requests_get, mocker):
    response = _make_mock_response(mocker, status_code=404)
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=response
    )
    mock_requests_get.return_value = response

    with pytest.raises(WebsiteExtractionError, match="404"):
        fetch_website_document(_TEST_URL)


def test_generic_request_exception_raises_clear_message(mock_requests_get):
    mock_requests_get.side_effect = requests.exceptions.RequestException("boom")

    with pytest.raises(WebsiteExtractionError, match="Something went wrong"):
        fetch_website_document(_TEST_URL)


# ---------------------------------------------------------------------------
# Content-level failures
# ---------------------------------------------------------------------------


def test_non_html_content_type_is_rejected(mock_requests_get, mocker):
    mock_requests_get.return_value = _make_mock_response(
        mocker, text="%PDF-1.4 ...", content_type="application/pdf"
    )

    with pytest.raises(WebsiteExtractionError, match="content type"):
        fetch_website_document(_TEST_URL)


def test_page_with_too_little_text_is_rejected(mock_requests_get, mocker):
    mock_requests_get.return_value = _make_mock_response(mocker, text=_THIN_HTML)

    with pytest.raises(WebsiteExtractionError, match="meaningful article content"):
        fetch_website_document(_TEST_URL)


def test_invalid_url_raises_url_validation_error_not_extraction_error(
    mock_requests_get,
):
    with pytest.raises(URLValidationError):
        fetch_website_document("not a url")

    mock_requests_get.assert_not_called()

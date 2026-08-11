"""
tests/test_llm.py

Tests for src/summarization/llm.py.

ChatGroq is fully mocked -- no real API key or network call is used.
groq's exception classes are constructed with real, valid arguments
(verified against the installed groq SDK directly, since several of
them require an httpx.Request/httpx.Response, not just a message
string), so these tests raise the actual exception types Groq would
raise, not simplified stand-ins.

get_llm() uses functools.lru_cache(maxsize=1), same pattern as
_get_embedding_model() in deduplicator.py -- so the cache is cleared
before and after every test to keep tests independent.
"""

import httpx
import groq
import pytest

from src.config import GROQ_MODEL_NAME
from src.summarization import llm
from src.summarization.llm import LLMGenerationError, get_llm, safe_invoke


@pytest.fixture(autouse=True)
def clear_llm_cache():
    llm.get_llm.cache_clear()
    yield
    llm.get_llm.cache_clear()


def _fake_request():
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _fake_response(status_code):
    return httpx.Response(status_code=status_code, request=_fake_request())


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------


def test_get_llm_constructs_chatgroq_with_configured_settings(mocker):
    mock_class = mocker.patch("src.summarization.llm.ChatGroq")
    mocker.patch("src.summarization.llm.get_groq_api_key", return_value="test-key")

    get_llm()

    mock_class.assert_called_once()
    _, kwargs = mock_class.call_args
    assert kwargs["model"] == GROQ_MODEL_NAME
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_retries"] == 2
    assert "api_key" in kwargs
    assert "timeout" in kwargs


def test_get_llm_is_constructed_only_once_across_multiple_calls(mocker):
    mock_class = mocker.patch("src.summarization.llm.ChatGroq")
    mocker.patch("src.summarization.llm.get_groq_api_key", return_value="test-key")

    get_llm()
    get_llm()
    get_llm()

    assert mock_class.call_count == 1


# ---------------------------------------------------------------------------
# safe_invoke -- success path
# ---------------------------------------------------------------------------


def test_safe_invoke_returns_chain_result_on_success(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.return_value = "A generated summary."

    result = safe_invoke(fake_chain, {"input": "some text"})

    assert result == "A generated summary."
    fake_chain.invoke.assert_called_once_with({"input": "some text"})


# ---------------------------------------------------------------------------
# safe_invoke -- each named Groq failure -> specific LLMGenerationError
# ---------------------------------------------------------------------------


def test_authentication_error_raises_clear_message(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = groq.AuthenticationError(
        "Invalid API key", response=_fake_response(401), body=None
    )

    with pytest.raises(LLMGenerationError, match="API key"):
        safe_invoke(fake_chain, {})


def test_rate_limit_error_raises_clear_message(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = groq.RateLimitError(
        "Rate limit exceeded", response=_fake_response(429), body=None
    )

    with pytest.raises(LLMGenerationError, match="rate limit"):
        safe_invoke(fake_chain, {})


def test_timeout_error_raises_clear_message(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = groq.APITimeoutError(request=_fake_request())

    with pytest.raises(LLMGenerationError, match="timed out"):
        safe_invoke(fake_chain, {})


def test_connection_error_raises_clear_message(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = groq.APIConnectionError(request=_fake_request())

    with pytest.raises(LLMGenerationError, match="connect"):
        safe_invoke(fake_chain, {})


def test_bad_request_error_raises_clear_message(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = groq.BadRequestError(
        "context_length_exceeded", response=_fake_response(400), body=None
    )

    with pytest.raises(LLMGenerationError, match="invalid"):
        safe_invoke(fake_chain, {})


def test_unnamed_status_error_is_caught_by_status_base_class(mocker):
    # PermissionDeniedError is deliberately NOT special-cased in
    # llm.py -- proves the APIStatusError catch-all still converts it
    # cleanly, and that the status code is included in the message.
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = groq.PermissionDeniedError(
        "Forbidden", response=_fake_response(403), body=None
    )

    with pytest.raises(LLMGenerationError, match="403"):
        safe_invoke(fake_chain, {})


def test_unnamed_api_error_is_caught_by_broadest_base_class(mocker):
    # A raw GroqError/APIError subclass with none of the more specific
    # shapes -- proves the final `except groq.APIError` catch-all works.
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = groq.APIError(
        "Unexpected failure", request=_fake_request(), body=None
    )

    with pytest.raises(LLMGenerationError, match="Something went wrong"):
        safe_invoke(fake_chain, {})


def test_non_groq_exceptions_are_not_caught(mocker):
    # safe_invoke only translates Groq's own exception hierarchy --
    # a totally unrelated bug (e.g. a KeyError in chain construction)
    # should propagate normally, not be silently swallowed or
    # mislabeled as a Groq failure.
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = KeyError("unexpected")

    with pytest.raises(KeyError):
        safe_invoke(fake_chain, {})

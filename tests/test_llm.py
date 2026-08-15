"""
tests/test_llm.py

Tests for src/summarization/llm.py.

ChatGroq is fully mocked -- no real API key or network call is used.
get_groq_api_key() is ALSO mocked in the get_llm() tests below -- this
is the fix for a real test-isolation gap: these tests previously
worked only because a local .env file happened to have SOME value for
GROQ_API_KEY, which is not something a test suite should depend on.
Mocking get_groq_api_key() directly means these tests behave
identically on any machine or CI runner, with or without a .env file.

groq's exception classes are constructed with real, valid arguments
(verified against the installed groq SDK directly).

get_llm() uses functools.lru_cache(maxsize=1), so the cache is cleared
before and after every test to keep tests independent.
"""

import httpx
import groq
import pytest

from src.config import GROQ_MODEL_NAME, GROQ_REQUEST_TIMEOUT_SECONDS, ConfigError
from src.summarization import llm
from src.summarization.llm import (
    LLMGenerationError,
    get_llm,
    safe_batch,
    safe_invoke,
    _MAX_RETRIES,
)


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
    mocker.patch(
        "src.summarization.llm.get_groq_api_key", return_value="fake-test-api-key"
    )
    mock_class = mocker.patch("src.summarization.llm.ChatGroq")

    get_llm()

    mock_class.assert_called_once()
    _, kwargs = mock_class.call_args
    assert kwargs["model"] == GROQ_MODEL_NAME
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_retries"] == _MAX_RETRIES
    assert kwargs["timeout"] == GROQ_REQUEST_TIMEOUT_SECONDS
    # Not asserting the exact api_key value/type here -- whether the
    # implementation passes the plain string or wraps it in
    # pydantic.SecretStr is an internal detail; what matters for this
    # test is that a key value is passed through at all.
    assert "api_key" in kwargs


def test_get_llm_is_constructed_only_once_across_multiple_calls(mocker):
    mocker.patch(
        "src.summarization.llm.get_groq_api_key", return_value="fake-test-api-key"
    )
    mock_class = mocker.patch("src.summarization.llm.ChatGroq")

    get_llm()
    get_llm()
    get_llm()

    assert mock_class.call_count == 1


def test_get_llm_raises_config_error_when_api_key_missing(mocker):
    # Formalizes the exact failure mode that exposed the test-isolation
    # gap in the first place: no GROQ_API_KEY configured anywhere
    # should surface as a clear ConfigError, and ChatGroq should never
    # even be constructed.
    mocker.patch(
        "src.summarization.llm.get_groq_api_key",
        side_effect=ConfigError(
            "GROQ_API_KEY is not set. Set it in a local .env file "
            "(see .env.example) or in Streamlit Cloud's secrets manager."
        ),
    )
    mock_class = mocker.patch("src.summarization.llm.ChatGroq")

    with pytest.raises(ConfigError, match="GROQ_API_KEY is not set"):
        get_llm()

    mock_class.assert_not_called()


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
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = groq.PermissionDeniedError(
        "Forbidden", response=_fake_response(403), body=None
    )

    with pytest.raises(LLMGenerationError, match="403"):
        safe_invoke(fake_chain, {})


def test_unnamed_api_error_is_caught_by_broadest_base_class(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = groq.APIError(
        "Unexpected failure", request=_fake_request(), body=None
    )

    with pytest.raises(LLMGenerationError, match="Something went wrong"):
        safe_invoke(fake_chain, {})


def test_non_groq_exceptions_are_not_caught_by_safe_invoke(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.invoke.side_effect = KeyError("unexpected")

    with pytest.raises(KeyError):
        safe_invoke(fake_chain, {})


# ---------------------------------------------------------------------------
# safe_batch -- success path
# ---------------------------------------------------------------------------


def test_safe_batch_returns_results_in_order(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.batch.return_value = ["summary A", "summary B", "summary C"]

    result = safe_batch(
        fake_chain,
        [{"chunk_content": "a"}, {"chunk_content": "b"}, {"chunk_content": "c"}],
    )

    assert result == ["summary A", "summary B", "summary C"]


def test_safe_batch_uses_default_max_concurrency(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.batch.return_value = []

    inputs = [{"chunk_content": "a"}, {"chunk_content": "b"}]
    safe_batch(fake_chain, inputs)

    fake_chain.batch.assert_called_once_with(inputs, config={"max_concurrency": 5})


def test_safe_batch_passes_through_custom_max_concurrency(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.batch.return_value = []

    safe_batch(fake_chain, [{"chunk_content": "a"}], max_concurrency=2)

    fake_chain.batch.assert_called_once_with(
        [{"chunk_content": "a"}], config={"max_concurrency": 2}
    )


# ---------------------------------------------------------------------------
# safe_batch -- Groq failures use the SAME shared translation as safe_invoke
# ---------------------------------------------------------------------------


def test_safe_batch_rate_limit_error_raises_clear_message(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.batch.side_effect = groq.RateLimitError(
        "Rate limit exceeded", response=_fake_response(429), body=None
    )

    with pytest.raises(LLMGenerationError, match="rate limit"):
        safe_batch(fake_chain, [{"chunk_content": "a"}])


def test_safe_batch_unnamed_status_error_is_caught_by_status_base_class(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.batch.side_effect = groq.PermissionDeniedError(
        "Forbidden", response=_fake_response(403), body=None
    )

    with pytest.raises(LLMGenerationError, match="403"):
        safe_batch(fake_chain, [{"chunk_content": "a"}])


def test_safe_batch_does_not_catch_non_groq_exceptions(mocker):
    fake_chain = mocker.MagicMock()
    fake_chain.batch.side_effect = KeyError("unexpected")

    with pytest.raises(KeyError):
        safe_batch(fake_chain, [{"chunk_content": "a"}])


def test_safe_invoke_and_safe_batch_produce_identical_messages_for_same_error(mocker):
    error = groq.RateLimitError(
        "Rate limit exceeded", response=_fake_response(429), body=None
    )

    invoke_chain = mocker.MagicMock()
    invoke_chain.invoke.side_effect = error

    batch_chain = mocker.MagicMock()
    batch_chain.batch.side_effect = error

    with pytest.raises(LLMGenerationError) as invoke_exc_info:
        safe_invoke(invoke_chain, {})
    with pytest.raises(LLMGenerationError) as batch_exc_info:
        safe_batch(batch_chain, [{}])

    assert invoke_exc_info.value.user_message == batch_exc_info.value.user_message

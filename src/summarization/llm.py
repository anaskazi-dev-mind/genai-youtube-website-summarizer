"""
src/summarization/llm.py

Wires up the Groq-backed LangChain chat model used by every
summarization strategy, and provides one centralized place that
translates Groq/network failures into this project's own exception
type with a user-safe message.

Includes exponential backoff retry logic for rate-limit errors,
allowing temporary quota exhaustion to recover without user intervention.
"""

import time
from functools import lru_cache
from typing import Any, List, Optional

import groq
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.config import (
    GROQ_MODEL_NAME,
    GROQ_REQUEST_TIMEOUT_SECONDS,
    GROQ_TEMPERATURE,
    GROQ_RATE_LIMIT_RETRY_ATTEMPTS,
    GROQ_RATE_LIMIT_RETRY_BASE_DELAY,
    get_groq_api_key,
)
from src.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 4
_DEFAULT_BATCH_MAX_CONCURRENCY = 5


class LLMGenerationError(RuntimeError):
    """
    Raised for any failure while calling Groq through LangChain.
    Carries a user-safe message (for the Streamlit UI) separately from
    the original exception (for logs).
    """

    def __init__(self, user_message: str, *, cause: Optional[Exception] = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.cause = cause


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Returns a single, process-wide ChatGroq instance, built from
    config.py's settings. Every summarization strategy calls this
    instead of constructing ChatGroq itself -- so changing the model,
    temperature, or timeout is always a one-line config.py change,
    never a change to any strategy's own code.
    """
    logger.info(
        "Initializing ChatGroq (model=%s, temperature=%s, timeout=%ss)",
        GROQ_MODEL_NAME,
        GROQ_TEMPERATURE,
        GROQ_REQUEST_TIMEOUT_SECONDS,
    )
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=SecretStr(get_groq_api_key()),
        temperature=GROQ_TEMPERATURE,
        timeout=GROQ_REQUEST_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )


def _translate_groq_exception(exc: Exception) -> LLMGenerationError:
    """
    Maps a caught groq exception to an LLMGenerationError with an
    appropriate user-safe message. Shared by safe_invoke and
    safe_batch so both call paths give identical messages for
    identical underlying failures.

    isinstance checks are ordered specific-to-general (mirroring how
    Python's own except-clause matching would order them), since this
    function receives an already-caught exception rather than relying
    on except-clause ordering itself.
    """
    if isinstance(exc, groq.AuthenticationError):
        return LLMGenerationError(
            "The Groq API key is invalid or missing. This is a setup "
            "issue, not a problem with your input -- check the "
            "GROQ_API_KEY configuration.",
            cause=exc,
        )
    if isinstance(exc, groq.RateLimitError):
        return LLMGenerationError(
            "Groq's rate limit was hit. The system has retried with exponential "
            "backoff. Please wait a moment and try again. Consider reducing content "
            "length or using a faster summarization strategy (Stuff).",
            cause=exc,
        )
    if isinstance(exc, groq.APITimeoutError):
        return LLMGenerationError(
            "The request to Groq timed out. This can happen with very "
            "large inputs -- try a shorter video or article, or try "
            "again.",
            cause=exc,
        )
    if isinstance(exc, groq.APIConnectionError):
        return LLMGenerationError(
            "Could not connect to Groq's API. Check your network "
            "connection and try again.",
            cause=exc,
        )
    if isinstance(exc, groq.BadRequestError):
        return LLMGenerationError(
            "Groq rejected this request as invalid -- this can happen "
            "if the input is too large for the model's context "
            "window, among other causes.",
            cause=exc,
        )
    if isinstance(exc, groq.APIStatusError):
        return LLMGenerationError(
            f"Groq returned an error (HTTP {exc.status_code}) while "
            "generating the summary.",
            cause=exc,
        )
    return LLMGenerationError(
        "Something went wrong while generating the summary with Groq.",
        cause=exc,
    )


def safe_invoke(chain: Runnable, chain_input: Any) -> Any:
    """
    Invokes any LangChain Runnable with exponential backoff retry on rate limits.

    Rate limit errors are retried with exponential backoff (e.g., 1s, 2s, 4s, 8s)
    up to GROQ_RATE_LIMIT_RETRY_ATTEMPTS times. Other errors are raised immediately
    with appropriate user-safe messages.

    This dramatically improves reliability on free Groq tier where quotas reset
    every minute but requests can exceed the limit.
    """
    for attempt in range(GROQ_RATE_LIMIT_RETRY_ATTEMPTS):
        try:
            return chain.invoke(chain_input)
        except groq.RateLimitError as exc:
            # Rate limit is temporary; retry with exponential backoff
            if attempt < GROQ_RATE_LIMIT_RETRY_ATTEMPTS - 1:
                wait_time = GROQ_RATE_LIMIT_RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %d seconds...",
                    attempt + 1,
                    GROQ_RATE_LIMIT_RETRY_ATTEMPTS,
                    wait_time,
                )
                time.sleep(wait_time)
                continue
            else:
                # All retries exhausted
                raise _translate_groq_exception(exc) from exc
        except groq.APIError as exc:
            # Other Groq errors are not retryable; fail immediately
            raise _translate_groq_exception(exc) from exc


def safe_batch(
    chain: Runnable,
    chain_inputs: List[Any],
    *,
    max_concurrency: int = _DEFAULT_BATCH_MAX_CONCURRENCY,
) -> List[Any]:
    """
    Runs a LangChain Runnable against multiple inputs CONCURRENTLY
    with exponential backoff retry on rate limits.

    Rate limit errors are retried with exponential backoff up to
    GROQ_RATE_LIMIT_RETRY_ATTEMPTS times. Other errors are raised immediately.

    Output order matches input order regardless of which call
    actually finishes first (verified directly).

    NOTE: With batch()'s default return_exceptions=False, ONE failing item
    aborts the entire batch. The exponential backoff retry mitigates this
    by retrying the entire batch if a rate limit is hit.
    """
    for attempt in range(GROQ_RATE_LIMIT_RETRY_ATTEMPTS):
        try:
            return chain.batch(
                chain_inputs, config={"max_concurrency": max_concurrency}
            )
        except groq.RateLimitError as exc:
            # Rate limit is temporary; retry with exponential backoff
            if attempt < GROQ_RATE_LIMIT_RETRY_ATTEMPTS - 1:
                wait_time = GROQ_RATE_LIMIT_RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Rate limit hit in batch (attempt %d/%d, %d items). "
                    "Retrying in %d seconds...",
                    attempt + 1,
                    GROQ_RATE_LIMIT_RETRY_ATTEMPTS,
                    len(chain_inputs),
                    wait_time,
                )
                time.sleep(wait_time)
                continue
            else:
                # All retries exhausted; raise after all attempts failed
                raise _translate_groq_exception(exc) from exc
        except groq.APIError as exc:
            # Other Groq errors are not retryable; fail immediately
            raise _translate_groq_exception(exc) from exc

    # This line should never be reached because the loop always either
    # returns, continues, or raises. But Pylance requires it for type safety.
    raise LLMGenerationError(
        "Unexpected error: batch processing completed without result or exception."
    )

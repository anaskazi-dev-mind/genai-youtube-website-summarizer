"""
src/summarization/llm.py

Wires up the Groq-backed LangChain chat model used by every
summarization strategy, and provides one centralized place that
translates Groq/network failures into this project's own exception
type with a user-safe message.

UPDATED for the Map-Reduce milestone: added safe_batch() for
concurrent per-chunk calls. The Groq exception-translation logic that
used to live only inside safe_invoke's except clauses is now a shared
helper, _translate_groq_exception(), used by BOTH safe_invoke and
safe_batch -- so the two call paths can never drift out of sync with
different messages for the same underlying Groq failure. safe_invoke's
external behavior is unchanged by this refactor.

Verified directly against langchain-groq 1.1.3 / groq SDK 0.37.1:
- ChatGroq's constructor args: `model` (alias model_name), `api_key`
  (alias groq_api_key). Constructing ChatGroq makes no network call.
- Runnable.batch() runs concurrently via a thread pool (config=
  {"max_concurrency": N}), preserves input order in its output, and
  raises the ORIGINAL exception type on failure -- not a wrapped one
  -- so the same exception handling applies to both invoke() and
  batch().
- groq's exception hierarchy: GroqError -> APIError -> (APIConnectionError
  -> APITimeoutError) and (APIStatusError -> RateLimitError,
  AuthenticationError, BadRequestError, PermissionDeniedError,
  NotFoundError, InternalServerError, ...).
"""

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
    get_groq_api_key,
)
from src.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 2
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
            "Groq's rate limit was hit for this request. Please wait "
            "a moment and try again.",
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
    Invokes any LangChain Runnable -- a bare ChatGroq instance, or a
    full `prompt | llm | parser` chain -- and translates every
    Groq/network failure into LLMGenerationError.
    """
    try:
        return chain.invoke(chain_input)
    except groq.APIError as exc:
        raise _translate_groq_exception(exc) from exc


def safe_batch(
    chain: Runnable,
    chain_inputs: List[Any],
    *,
    max_concurrency: int = _DEFAULT_BATCH_MAX_CONCURRENCY,
) -> List[Any]:
    """
    Runs a LangChain Runnable against multiple inputs CONCURRENTLY
    (via Runnable.batch()'s thread pool), and translates every
    Groq/network failure into LLMGenerationError, exactly like
    safe_invoke.

    Output order matches input order regardless of which call
    actually finishes first (verified directly).

    NOTE, an honest limitation: with batch()'s default
    return_exceptions=False, ONE failing item aborts the entire batch
    -- there's no partial-results fallback where successfully
    summarized chunks are kept and only the failed one is retried.
    That's a real trade-off, not something this version handles.
    """
    try:
        return chain.batch(chain_inputs, config={"max_concurrency": max_concurrency})
    except groq.APIError as exc:
        raise _translate_groq_exception(exc) from exc

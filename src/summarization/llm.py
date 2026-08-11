"""
src/summarization/llm.py

Wires up the Groq-backed LangChain chat model used by every
summarization strategy, and provides one centralized place that
translates Groq/network failures into this project's own exception
type with a user-safe message -- mirrors the pattern already used in
youtube_extractor.py and website_extractor.py.

Verified directly against langchain-groq 1.1.3 and the underlying groq
SDK 0.37.1 rather than assumed:
- ChatGroq's constructor argument for the model name is `model` (an
  alias for the internal field `model_name`), and for the API key is
  `api_key` (alias for `groq_api_key`) -- matching the same convention
  other LangChain chat model integrations use.
- Constructing ChatGroq does NOT make a network call -- confirmed by
  successfully instantiating it with an invalid key. Failures only
  surface on the first real .invoke() call.
- groq's exception hierarchy: GroqError -> APIError -> (APIConnectionError
  -> APITimeoutError) and (APIStatusError -> RateLimitError,
  AuthenticationError, BadRequestError, PermissionDeniedError,
  NotFoundError, InternalServerError, ...).
"""

from pydantic import SecretStr
from functools import lru_cache
from typing import Any, Optional

import groq
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from src.config import (
    DEFAULT_GROQ_MODEL,
    GROQ_MODEL_NAME,
    GROQ_REQUEST_TIMEOUT_SECONDS,
    GROQ_TEMPERATURE,
    get_groq_api_key,
)
from src.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 2


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
    never a change to Stuff/Map-Reduce/Refine's own code.
    """
    logger.info(
        "Initializing ChatGroq (model=%s, temperature=%s, timeout=%ss)",
        GROQ_MODEL_NAME,
        GROQ_TEMPERATURE,
        GROQ_REQUEST_TIMEOUT_SECONDS,
    )
    chat_groq_cls: Any = ChatGroq
    return chat_groq_cls(
        model=GROQ_MODEL_NAME or DEFAULT_GROQ_MODEL,
        api_key=SecretStr(get_groq_api_key()),
        temperature=GROQ_TEMPERATURE,
        timeout=GROQ_REQUEST_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )


def safe_invoke(chain: Runnable, chain_input: Any) -> Any:
    """
    Invokes any LangChain Runnable -- a bare ChatGroq instance, or a
    full `prompt | llm | parser` chain -- and translates every
    Groq/network failure into LLMGenerationError with a message
    appropriate to show directly in the Streamlit UI.

    Centralizing this here means every summarization strategy (Stuff,
    Map-Reduce, Refine) gets identical, single-tested error handling
    instead of each strategy file reimplementing its own try/except
    around Groq calls.

    Exception ordering matters: more specific exceptions are caught
    before their broader parent classes (e.g. AuthenticationError
    before its parent APIStatusError), otherwise Python would always
    match the broader except block first and the specific message
    would never be reached.
    """
    try:
        return chain.invoke(chain_input)
    except groq.AuthenticationError as exc:
        raise LLMGenerationError(
            "The Groq API key is invalid or missing. This is a setup "
            "issue, not a problem with your input -- check the "
            "GROQ_API_KEY configuration.",
            cause=exc,
        ) from exc
    except groq.RateLimitError as exc:
        raise LLMGenerationError(
            "Groq's rate limit was hit for this request. Please wait "
            "a moment and try again.",
            cause=exc,
        ) from exc
    except groq.APITimeoutError as exc:
        raise LLMGenerationError(
            "The request to Groq timed out. This can happen with very "
            "large inputs -- try a shorter video or article, or try "
            "again.",
            cause=exc,
        ) from exc
    except groq.APIConnectionError as exc:
        raise LLMGenerationError(
            "Could not connect to Groq's API. Check your network "
            "connection and try again.",
            cause=exc,
        ) from exc
    except groq.BadRequestError as exc:
        raise LLMGenerationError(
            "Groq rejected this request as invalid -- this can happen "
            "if the input is too large for the model's context "
            "window, among other causes.",
            cause=exc,
        ) from exc
    except groq.APIStatusError as exc:
        # Catch-all for any other named HTTP-status failure (e.g.
        # InternalServerError, PermissionDeniedError, NotFoundError)
        # not special-cased above.
        raise LLMGenerationError(
            f"Groq returned an error (HTTP {exc.status_code}) while "
            "generating the summary.",
            cause=exc,
        ) from exc
    except groq.APIError as exc:
        # Broadest catch-all in the Groq SDK's exception hierarchy --
        # anything not covered by a more specific case above.
        raise LLMGenerationError(
            "Something went wrong while generating the summary with Groq.",
            cause=exc,
        ) from exc

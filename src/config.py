"""
src/config.py

Centralized configuration for the AI YouTube & Website Summarizer.

Single source of truth for:
- API keys / secrets (loaded from environment variables or Streamlit secrets)
- Model configuration (which Groq model, which HuggingFace embedding model)
- Chunking parameters (chunk size, chunk overlap)
- Deduplication threshold
- Logging level

Nothing outside this file should call os.environ or st.secrets directly.
That keeps "where do we get GROQ_API_KEY from" a one-file answer, and
means swapping the model or tuning chunk size never touches business logic.
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load .env into os.environ for local development.
# On Streamlit Cloud there is no .env file, so this is a harmless no-op;
# secrets are provided via Streamlit's secrets manager instead (see
# _get_secret below).
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing."""


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Look up a configuration value from, in order:
      1. Environment variables. This covers local .env files AND
         Streamlit Cloud, because Streamlit automatically mirrors
         root-level secrets.toml entries into os.environ.
      2. st.secrets directly, as a defensive fallback.
      3. The provided default.

    We check os.environ first (not st.secrets first) so reading config
    never requires importing streamlit or running under `streamlit run`
    -- useful for plain pytest runs and standalone scripts.
    """
    value = os.environ.get(key)
    if value:
        return value

    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        # Not running inside Streamlit, or no secrets.toml configured.
        # Expected in local pytest runs -- not an error.
        pass

    return default


# ---------------------------------------------------------------------------
# Groq / LLM configuration
# ---------------------------------------------------------------------------


def get_groq_api_key() -> str:
    """
    Returns the Groq API key, or raises ConfigError with a clear,
    non-sensitive message if it isn't configured anywhere.

    This is a function, not a module-level constant, so importing this
    module never fails just because a key isn't set yet -- the error
    only surfaces when something actually tries to call Groq.
    """
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        raise ConfigError(
            "GROQ_API_KEY is not set. Set it in a local .env file "
            "(see .env.example) or in Streamlit Cloud's secrets manager."
        )
    return api_key


# Default Groq model. Full reasoning (context window, cost, why the
# deprecated llama-3.3-70b-versatile was avoided) lives in
# docs/ARCHITECTURE.md. Overridable via GROQ_MODEL_NAME so a future
# Groq deprecation is a one-variable change, not a code change.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL_NAME = _get_secret("GROQ_MODEL_NAME", DEFAULT_GROQ_MODEL)

# Low temperature reduces creative drift -- matters for factual summarization.
GROQ_TEMPERATURE = 0.2

# Map-Reduce/Refine mean many sequential Groq calls for long inputs; we'd
# rather fail one call clearly than hang the whole Streamlit session.
GROQ_REQUEST_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# HuggingFace configuration
# ---------------------------------------------------------------------------

# Small (~90MB), CPU-friendly sentence-embedding model used only for
# near-duplicate chunk filtering before summarization -- not for
# generating any part of the summary itself. Chosen because it runs
# locally without a GPU or API token, which matters for Streamlit
# Cloud's free tier. Alternatives we rejected (a local HF summarization
# model as a "fallback") are documented in docs/ARCHITECTURE.md.
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine-similarity threshold above which two chunks are treated as
# near-duplicates. This is a starting value -- we'll validate and
# justify it with real chunks when we implement deduplicator.py, not
# leave it as an unexamined guess.
DEDUP_SIMILARITY_THRESHOLD = 0.92

# ---------------------------------------------------------------------------
# Website extraction configuration
# ---------------------------------------------------------------------------

# Minimum amount of extracted text required for a webpage to be considered
# meaningful content. Pages below this threshold may be empty, JS-rendered,
# or mostly boilerplate.
MIN_ACCEPTABLE_TEXT_LENGTH = 200

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

# Starting values -- the full reasoning (why this size, what happens if
# it's too small/large) is documented when we implement chunker.py,
# where we can reason about it against real transcripts/articles.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = _get_secret("LOG_LEVEL", "INFO") or "INFO"

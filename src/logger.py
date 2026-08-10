"""
src/logger.py

Centralized logging setup for the AI YouTube & Website Summarizer.

Goals for this logger (from the project brief):
- Help answer: what source was processed, which strategy was selected,
  how many documents/chunks were generated, and where processing failed.
- Never spam: one line per meaningful pipeline event, not per-token or
  per-request noise.
- Never log secrets: API keys must never appear in a log line, even at
  DEBUG level.

Streamlit Cloud captures stdout/stderr into its own log viewer, so we
log to the console (StreamHandler) rather than writing to a local file
-- a file on Streamlit Cloud's filesystem is ephemeral and wouldn't be
visible anywhere useful anyway.
"""

import logging
import sys

from src.config import LOG_LEVEL

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_once() -> None:
    """
    Attaches a single console handler to the root logger, exactly once.
    Guards against duplicate handlers (and duplicate log lines) since
    get_logger() will be called from many modules across one process.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.addHandler(handler)

    # Quiet down noisy third-party loggers we don't control and don't
    # want flooding our own pipeline logs (HTTP client chatter, etc.).
    for noisy_logger in ("urllib3", "httpx", "httpcore", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a module-scoped logger, e.g. get_logger(__name__).
    Safe to call from every module -- root handler setup only happens once.
    """
    _configure_root_once()
    return logging.getLogger(name)


def truncate_for_log(text: str, max_length: int = 200) -> str:
    """
    Returns a short preview of text safe for logging -- e.g. the first
    200 characters of an extracted transcript -- instead of dumping
    entire documents into the log stream. Used anywhere we want to
    confirm "we got content" without spamming the logs.
    """
    if text is None:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}... [{len(text)} chars total]"

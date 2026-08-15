"""
app.py

Streamlit entry point. Contains ONLY UI logic -- collecting input,
calling the pipeline (extraction -> cleaning -> chunking -> dedup ->
summarization), and rendering results or errors. No business logic
lives here; everything is delegated to src/.
"""

import streamlit as st

from src.config import ConfigError
from src.extractors.website_extractor import (
    WebsiteExtractionError,
    fetch_website_document,
)
from src.extractors.youtube_extractor import (
    YouTubeExtractionError,
    fetch_youtube_document,
)
from src.logger import get_logger
from src.processing.chunker import split_documents
from src.processing.cleaner import clean_document
from src.processing.deduplicator import deduplicate_chunks
from src.summarization.base_strategy import SummarizationError, SummaryResult
from src.summarization.llm import LLMGenerationError
from src.summarization.strategy_factory import available_strategies, get_strategy
from src.validators import (
    SourceType,
    URLValidationError,
    detect_source_type,
    is_valid_url,
)

logger = get_logger(__name__)

st.set_page_config(
    page_title="AI YouTube & Website Summarizer",
    page_icon="📝",
    layout="centered",
)

# Maps internal strategy names (from strategy_factory.available_strategies())
# to a friendlier label for the dropdown. If a new strategy is added and
# this dict isn't updated, format_func just falls back to the raw name --
# nothing breaks, it's just slightly less pretty.
_STRATEGY_DISPLAY_NAMES = {
    "stuff": "Stuff -- fast, best for short content",
    "map_reduce": "Map-Reduce -- best for long content",
    "refine": "Refine -- sequential, best for narrative continuity",
}

# Exceptions that carry a pre-written, safe-to-display .user_message --
# every extraction/summarization/LLM error type in this project follows
# this convention (see the corresponding milestones).
_USER_FACING_ERRORS = (
    URLValidationError,
    YouTubeExtractionError,
    WebsiteExtractionError,
    SummarizationError,
    LLMGenerationError,
)


def _init_session_state() -> None:
    """
    Streamlit reruns this entire script on every interaction. Without
    persisting the result (and any error) in session_state, an
    unrelated rerun -- e.g. expanding the "Source details" section --
    would make the summary disappear, since it would only exist for
    the single rerun triggered by the Summarize button itself.

    Uses atomic updates via dict.update() to minimize race conditions
    in Streamlit's concurrent rerun model.
    """
    defaults = {
        "summary_result": None,
        "error_message": None,
        # Bumped on "Start Over" to force the URL text_input to reset --
        # changing a widget's key is the standard Streamlit pattern for
        # clearing a widget's value, since a widget's value can't be
        # reassigned directly after it's been instantiated.
        "input_key_version": 0,
        "pipeline_running": False,
    }
    # Atomic update: only set keys that don't already exist
    for key, default_value in defaults.items():
        st.session_state.setdefault(key, default_value)


def _run_pipeline(url: str, strategy_name: str, status) -> SummaryResult:
    """
    Runs the full pipeline for one URL, updating the given st.status
    container as it progresses through each documented pipeline stage.
    """
    source_type = detect_source_type(url)

    status.update(label="Extracting content...")
    if source_type == SourceType.YOUTUBE:
        document = fetch_youtube_document(url)
    else:
        document = fetch_website_document(url)
    logger.info("Extracted document from %s (source_type=%s)", url, source_type.value)

    status.update(label="Cleaning text...")
    document = clean_document(document)

    status.update(label="Splitting into chunks...")
    chunks = split_documents([document])

    status.update(label=f"Checking {len(chunks)} chunk(s) for redundant content...")
    chunks = deduplicate_chunks(chunks)

    # ISSUE FIX #5: Validate that deduplication didn't remove all chunks
    if not chunks:
        raise SummarizationError(
            "All content chunks were identified as near-duplicates and removed. "
            "This typically happens with highly repetitive content. "
            "Please try a different video or article."
        )

    status.update(
        label=f"Summarizing {len(chunks)} chunk(s) with the "
        f"'{strategy_name}' strategy..."
    )
    strategy = get_strategy(strategy_name)
    return strategy.summarize(chunks)


def _render_result(result: SummaryResult) -> None:
    st.markdown(result.content)

    with st.expander("Source details"):
        metadata = result.source_metadata
        st.write(f"**Source type:** {metadata.get('source_type', 'unknown').title()}")
        if metadata.get("title"):
            st.write(f"**Title:** {metadata['title']}")
        if metadata.get("source_url"):
            st.write(f"**URL:** {metadata['source_url']}")
        st.write(f"**Strategy used:** {result.strategy}")
        st.write(f"**Chunks processed:** {result.chunk_count}")


def _get_rate_limit_help() -> str:
    """Returns helpful guidance when rate limited."""
    return """
    ### Groq API Rate Limit Exceeded

    You've hit Groq's free tier rate limit. Here's what to do:

    1. **Wait 1-2 minutes** - Groq quotas reset automatically. Then try again.
    2. **Use Stuff strategy** - Summarizes everything in 1 API call instead of many.
    3. **Try shorter content** - Fewer chunks = fewer API calls.
    4. **Consider Groq Paid Plan** - Free tier has ~30 requests/minute. Paid tiers are much higher.

    The system already retried with exponential backoff, so this is a genuine limit.
    """


def main() -> None:
    _init_session_state()

    st.title("📝 AI YouTube & Website Summarizer")
    st.caption(
        "Paste a YouTube video link or a website/article URL, choose a "
        "summarization strategy, and get a structured summary."
    )

    url = st.text_input(
        "YouTube video or website URL",
        placeholder="https://www.youtube.com/watch?v=... or https://example.com/article",
        key=f"url_input_{st.session_state.input_key_version}",
    )

    if url.strip() and is_valid_url(url):
        detected = detect_source_type(url)
        label = (
            "YouTube video" if detected == SourceType.YOUTUBE else "Website / article"
        )
        st.caption(f"Detected source: **{label}**")

    strategy_name = st.selectbox(
        "Summarization strategy",
        options=available_strategies(),
        format_func=lambda name: str(_STRATEGY_DISPLAY_NAMES.get(name, name)),
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        summarize_clicked = st.button(
            "Summarize", type="primary", disabled=not url.strip()
        )
    with col2:
        if st.session_state.summary_result is not None:
            if st.button("Start Over"):
                st.session_state.summary_result = None
                st.session_state.error_message = None
                st.session_state.input_key_version += 1
                st.rerun()

    if summarize_clicked:
        st.session_state.summary_result = None
        st.session_state.error_message = None

        if not is_valid_url(url):
            st.session_state.error_message = "Please enter a valid http(s) URL."
        else:
            with st.status("Starting...", expanded=True) as status:
                try:
                    result = _run_pipeline(url.strip(), strategy_name, status)
                    status.update(
                        label="Done!",
                        state="complete",
                        expanded=False,
                    )
                    st.session_state.summary_result = result

                except LLMGenerationError as exc:
                    status.update(
                        label="Failed",
                        state="error",
                        expanded=True,
                    )
                    # Check if it's a rate limit error and provide extra help
                    if "rate limit" in str(exc).lower():
                        st.session_state.error_message = str(exc)
                        st.markdown(_get_rate_limit_help())
                    else:
                        st.session_state.error_message = str(exc)
                    logger.info(
                        "Handled LLM error for %s: %s",
                        url,
                        str(exc),
                    )

                except _USER_FACING_ERRORS as exc:
                    status.update(
                        label="Failed",
                        state="error",
                        expanded=True,
                    )
                    st.session_state.error_message = str(exc)
                    logger.info(
                        "Handled pipeline error for %s: %s",
                        url,
                        str(exc),
                    )

                except ConfigError as exc:
                    status.update(
                        label="Configuration error",
                        state="error",
                        expanded=True,
                    )
                    st.session_state.error_message = str(exc)

                except Exception:
                    status.update(
                        label="Failed",
                        state="error",
                        expanded=True,
                    )
                    st.session_state.error_message = (
                        "An unexpected error occurred. Please try again -- "
                        "if this keeps happening, this content may not be "
                        "supported yet."
                    )
                    logger.exception(
                        "Unexpected error while summarizing %s",
                        url,
                    )
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    if st.session_state.summary_result is not None:
        _render_result(st.session_state.summary_result)


if __name__ == "__main__":
    main()

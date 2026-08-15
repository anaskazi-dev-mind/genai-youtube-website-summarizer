"""
src/summarization/prompts.py

Every prompt used by the three summarization strategies lives here --
nowhere else in the codebase constructs a prompt string. This keeps
prompt engineering changes isolated to one file, and means the
structured-summary format is defined exactly ONCE and shared, instead
of being redefined (and risking drift) in three separate strategy files.

Verified directly: ChatPromptTemplate.from_messages([("system", ...),
("human", ...)]) correctly detects {variable} placeholders and produces
SystemMessage/HumanMessage pairs in order (not assumed from memory).
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Shared building blocks -- defined once, reused across every FINAL-output
# prompt (Stuff, the Reduce step, and Refine's update step), so the
# structured summary format can never silently drift between strategies.
# ---------------------------------------------------------------------------

_STRUCTURED_SUMMARY_FORMAT = """
Format your response using EXACTLY this structure, in Markdown:

## Title
A concise, descriptive title for the content (max 12 words).

## Executive Summary
2-3 sentences capturing the core message.

## Key Points
- 4-6 bullet points, each a single self-contained idea.

## Important Details
- 2-5 bullet points with supporting facts, figures, or examples worth remembering.

## Main Takeaways
- 2-4 bullet points on what the reader or viewer should walk away understanding.

## Conclusion
1-2 sentences wrapping up the overall message.

Do not include any text before "## Title" or after the Conclusion section.
""".strip()

_FACTUALITY_CONSTRAINTS = """
Constraints:
- Base the summary ONLY on the provided content. Do not add outside knowledge, opinions, or speculation.
- If the content does not contain enough information for a section, write "Not enough information in the source content" for that section instead of inventing details.
- Do not mention that you are an AI, or that this is a summary of a transcript or article -- write directly about the subject matter.
- Preserve factual accuracy over stylistic flourish. Do not exaggerate claims made in the source.
""".strip()


# ---------------------------------------------------------------------------
# Stuff strategy: entire content fits in one call.
# Objective: produce the final structured summary directly from the full
# content in a single pass.
# ---------------------------------------------------------------------------

_STUFF_SYSTEM_PROMPT = f"""
You are an expert content summarizer. You will be given the full text of
either a YouTube video transcript or a website article. Produce a
structured summary of it.

{_FACTUALITY_CONSTRAINTS}

{_STRUCTURED_SUMMARY_FORMAT}
""".strip()

STUFF_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _STUFF_SYSTEM_PROMPT),
        ("human", "{content}"),
    ]
)


# ---------------------------------------------------------------------------
# Map-Reduce, map step: summarize ONE chunk in isolation.
# Objective: produce a short, dense, PLAIN-TEXT (not structured) summary
# of a single chunk -- this is an internal intermediate artifact, never
# shown to the user directly, so it deliberately does NOT use the
# structured format. Forcing every chunk into six Markdown sections would
# waste tokens on structure that gets discarded once the Reduce step
# combines everything into one final structured summary anyway.
# ---------------------------------------------------------------------------

_MAP_SYSTEM_PROMPT = """
You are summarizing one chunk of a larger video transcript or article.

This is an intermediate summary that will later be merged with summaries
from other chunks.

Rules:
- Maximum 3 bullet points.
- Maximum 40 words total.
- Preserve only the most important facts, names, dates, numbers and key claims.
- Remove repetition, examples, filler and conversational text.
- Do not use Markdown headings.
- Base the summary only on the provided content.
- Do not add outside knowledge or speculation.
""".strip()

MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _MAP_SYSTEM_PROMPT),
        ("human", "{chunk_content}"),
    ]
)


# ---------------------------------------------------------------------------
# Map-Reduce, reduce step: combine the per-chunk summaries into the final
# structured summary.
# ---------------------------------------------------------------------------

_REDUCE_SYSTEM_PROMPT = f"""
You will be given a series of section summaries, in their original
order, from a single video transcript or article that was too long to
process in one pass. Synthesize them into ONE coherent structured
summary of the entire content -- do not just concatenate the section
summaries or summarize them individually.

{_FACTUALITY_CONSTRAINTS}

{_STRUCTURED_SUMMARY_FORMAT}
""".strip()

REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _REDUCE_SYSTEM_PROMPT),
        ("human", "Section summaries, in order:\n\n{combined_summaries}"),
    ]
)


# ---------------------------------------------------------------------------
# Refine, initial step: build the first version of the summary from the
# first chunk only.
# ---------------------------------------------------------------------------

_REFINE_INITIAL_SYSTEM_PROMPT = f"""
You are an expert content summarizer. You will be given the FIRST
section of a longer video transcript or article -- more sections will
follow, and this summary will be refined as they arrive. Produce a
structured summary based on what you have so far.

{_FACTUALITY_CONSTRAINTS}

{_STRUCTURED_SUMMARY_FORMAT}
""".strip()

REFINE_INITIAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _REFINE_INITIAL_SYSTEM_PROMPT),
        ("human", "{first_chunk_content}"),
    ]
)


# ---------------------------------------------------------------------------
# Refine, update step: given the existing summary and the next chunk,
# produce an updated summary. Deliberately outputs the FULL updated
# summary each time (not a diff/patch) -- diffing text reliably is a
# harder, riskier problem than just asking the model to rewrite the whole
# thing with the new information folded in, and downstream code only ever
# needs to store one current summary, not a summary plus a changelog.
# ---------------------------------------------------------------------------

_REFINE_UPDATE_SYSTEM_PROMPT = f"""
You are refining an existing structured summary with new information.
You will be given the EXISTING summary and NEW content that comes
later in the same video transcript or article.

Update the existing summary to incorporate any new, genuinely
significant information from the new content:
- Keep existing points that are still accurate.
- Add new key points or details only if they meaningfully add to the
  summary -- do not add filler just because new text was provided.
- If the new content contradicts something in the existing summary,
  prefer the new content, since it reflects more complete information.
- If the new content doesn't add anything meaningful, output the
  existing summary largely unchanged.

Output the COMPLETE updated summary in the format below -- not a list
of changes.

{_FACTUALITY_CONSTRAINTS}

{_STRUCTURED_SUMMARY_FORMAT}
""".strip()

REFINE_UPDATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _REFINE_UPDATE_SYSTEM_PROMPT),
        (
            "human",
            "EXISTING SUMMARY:\n{existing_summary}\n\nNEW CONTENT:\n{new_content}",
        ),
    ]
)

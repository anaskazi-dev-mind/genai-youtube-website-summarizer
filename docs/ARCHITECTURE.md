# Architecture Deep Dive — AI YouTube & Website Summarizer

This document provides a detailed overview of the project's internal architecture and the reasoning behind its design decisions. While the README explains **what** the project does, this document focuses on **how** it works and **why** specific implementation choices were made.

It also documents important engineering decisions, trade-offs, and lessons learned during development, including cases where the original approach had to be revised after validating real library behavior.

---

# 1. Design Philosophy

Three core principles guided every architectural decision in this project.

## 1.1 Verify, Don't Assume

Every third-party library used in this project was installed, tested, and inspected before being integrated into the application.

This includes:

- `youtube-transcript-api`
- `trafilatura`
- `langchain-groq`
- `sentence-transformers`
- `groq`
- `langchain-text-splitters`

Instead of relying on tutorials or outdated examples, the implementation was verified against the actual library APIs by checking:

- Constructor signatures
- Exception hierarchies
- Runtime behavior
- Official documentation
- Source code when necessary

For example, `Runnable.batch()` was verified to execute requests concurrently instead of assuming its behavior from documentation alone.

---

## 1.2 Fail Clearly Instead of Failing Silently

Whenever a known failure occurs, the application raises a project-specific exception with a clear, user-friendly message.

The project avoids:

- Silent failures
- Hidden fallback behavior
- Unexpected default models
- Swallowed exceptions

Each known failure scenario is handled explicitly so users understand what went wrong and how to resolve it.

---

## 1.3 Document Limitations Honestly

Not every implementation decision is based on benchmarked research.

Some values are practical engineering defaults, including:

- Chunk size
- Chunk overlap
- Deduplication similarity threshold
- Maximum safe size for the Stuff strategy

Instead of presenting these values as optimal, the project documents them as reasoned implementation choices and clearly states their limitations.

---

# 2. End-to-End Pipeline

The application follows a modular processing pipeline where each stage performs a single responsibility before passing the result to the next stage.

This separation makes every component independently testable, reusable, and easier to maintain.

## 2.1 YouTube Processing Pipeline

```
Input URL
    │
    ▼
URL Validation
(src/validators.py)
    │
    ▼
Transcript Extraction
(src/extractors/youtube_extractor.py)
    │
    ▼
Document Cleaning
(src/processing/cleaner.py)
    │
    ▼
Text Chunking
(src/processing/chunker.py)
    │
    ▼
Semantic Deduplication
(src/processing/deduplicator.py)
    │
    ▼
Summarization Strategy
(src/summarization/*)
    │
    ▼
SummaryResult
    │
    ▼
Streamlit UI (app.py)
```

### Step 1 — URL Validation

**File:** `src/validators.py`

The pipeline begins by validating the supplied YouTube URL.

The validator supports multiple YouTube URL formats, including:

- `watch?v=`
- `youtu.be/`
- `/embed/`
- `/shorts/`
- `/live/`

A regular-expression and `urllib` based parser extracts the standard 11-character video ID.

If no valid ID can be extracted, the validator raises a `URLValidationError`.

---

### Step 2 — Transcript Extraction

**File:** `src/extractors/youtube_extractor.py`

The application retrieves transcripts using the current instance-based API:

```python
YouTubeTranscriptApi().fetch(...)
```

The extractor requests English transcripts using:

```python
languages=("en", "en-US", "en-GB")
```

Every named exception from the library's `CouldNotRetrieveTranscript` hierarchy is translated into a project-specific `YouTubeExtractionError` with a user-friendly message.

On success, the extractor returns a single LangChain `Document` containing metadata such as:

- `source_type`
- `video_id`
- `source_url`
- `language`
- `is_generated`

---

### Step 3 — Document Cleaning

**File:** `src/processing/cleaner.py`

Before chunking, the transcript is normalized and cleaned.

The cleaning stage performs:

- Unicode NFKC normalization
- Non-breaking space replacement
- Removal of transcript annotations (e.g. `[Music]`, `[Applause]`)
- Whitespace normalization

This produces cleaner text for downstream processing.

---

### Step 4 — Text Chunking

**File:** `src/processing/chunker.py`

Long documents are divided using LangChain's `RecursiveCharacterTextSplitter`.

Configuration:

- `CHUNK_SIZE = 4000`
- `CHUNK_OVERLAP = 400`

Each generated chunk also receives metadata including:

- `chunk_index`
- `start_index`

This metadata is preserved throughout the summarization pipeline.

---

### Step 5 — Semantic Deduplication

**File:** `src/processing/deduplicator.py`

To reduce redundant processing, semantically similar chunks are filtered before being sent to the LLM.

The implementation uses:

- `sentence-transformers/all-MiniLM-L6-v2`
- Greedy **O(n²)** similarity comparison
- Cosine similarity threshold of **0.92**
- Order-preserving duplicate removal

This reduces unnecessary LLM calls while maintaining the original document flow.

---

### Step 6 — Summarization

**Directory:** `src/summarization/`

The processed chunks are passed to the selected summarization strategy.

The project currently supports:

- Stuff
- Map-Reduce
- Refine

Each strategy implements the common `SummarizationStrategy` interface and returns a standardized `SummaryResult`.

---

### Step 7 — UI Rendering

**File:** `app.py`

Finally, the generated `SummaryResult` is rendered in the Streamlit interface using a consistent Markdown format.

The UI layer contains presentation logic only, while all business logic remains inside the `src/` package.

### 2.2 Website Path

The website pipeline is identical to the YouTube pipeline from
`clean_document()` onward. The only difference lies in the content
extraction stage.

```
URL
↓
is_valid_url()
↓
fetch_website_document()
[src/extractors/website_extractor.py]
```

**Extraction process:**

- Fetches the webpage using `requests.get()` with a real User-Agent
  and a 15-second timeout.
- Validates the response by checking the `Content-Type`; non-HTML
  responses are rejected.
- Attempts content extraction using:

  ```python
  trafilatura.extract(
      html,
      output_format="json",
      with_metadata=True
  )
  ```

- If `trafilatura` extracts fewer than **200 characters**, the
  application automatically falls back to a BeautifulSoup-based
  extraction by:
  - removing `script`, `style`, `nav`, `header`, `footer`,
    `noscript`, and `form` tags
  - extracting and joining the remaining `<p>` elements

- Returns a LangChain `Document` containing metadata such as:
  - `source_type`
  - `source_url`
  - `title`

Once extraction is complete, both YouTube and website inputs are
converted into the same `Document` format. This common interface keeps
all downstream stages—cleaning, chunking, deduplication, and
summarization—completely source-agnostic.

---

## 3. Module Responsibilities

| Module | Responsibility | Depends on |
|---|---|---|
| `src/config.py` | Central configuration for secrets and application settings. The only module that reads `os.environ` or `st.secrets` directly. | `python-dotenv`, `streamlit` (optional) |
| `src/logger.py` | Configures the root logger and provides reusable loggers through `get_logger(__name__)`. | stdlib `logging` |
| `src/validators.py` | URL validation, source detection, and YouTube video ID extraction. Contains only pure functions with no I/O. | stdlib `urllib.parse`, `re` |
| `src/extractors/youtube_extractor.py` | Retrieves YouTube transcripts and translates library exceptions into project-specific errors. | `youtube-transcript-api`, `langchain-core` |
| `src/extractors/website_extractor.py` | Fetches webpages, extracts main content, and handles extraction-related errors. | `requests`, `trafilatura`, `beautifulsoup4`, `langchain-core` |
| `src/processing/cleaner.py` | Performs source-agnostic text normalization and cleanup. | stdlib `re`, `unicodedata` |
| `src/processing/chunker.py` | Splits documents into overlapping chunks and estimates token boundaries. | `langchain-text-splitters` |
| `src/processing/deduplicator.py` | Detects and removes semantically similar chunks using embeddings. | `sentence-transformers` |
| `src/summarization/prompts.py` | Stores every prompt template used throughout the application. | `langchain-core` |
| `src/summarization/llm.py` | Creates the Groq client and translates API exceptions through `safe_invoke()` and `safe_batch()`. | `langchain-groq`, `groq` |
| `src/summarization/base_strategy.py` | Defines the common strategy interface, `SummaryResult`, and shared helper methods. | `langchain-core` |
| `src/summarization/{stuff,map_reduce,refine}_strategy.py` | Implements the individual summarization strategies. | `llm.py`, `prompts.py`, `base_strategy.py` |
| `src/summarization/strategy_factory.py` | Maps strategy names to their corresponding implementation classes. | Strategy modules |
| `app.py` | Handles the Streamlit UI, progress display, session state, and result rendering. | `streamlit`, all modules in `src/` |

The dependency flow is intentionally one-directional.

```
app.py
        ↓
src/summarization/
        ↓
src/processing/
        ↓
prompts.py / llm.py
        ↓
config.py / logger.py
```

No module inside `src/` imports anything from `app.py`, keeping the
business logic completely independent from the user interface.

---

## 4. Design Patterns Used

### Strategy Pattern

Implemented using `base_strategy.py`, the three strategy classes, and
`strategy_factory.py`.

Each summarization algorithm follows the same interface and can be
selected dynamically at runtime. Adding a new strategy only requires
creating one new strategy file and registering it in the factory.

### Factory Pattern

Implemented in `strategy_factory.get_strategy()`.

This centralizes the mapping between a strategy name and its concrete
implementation, avoiding repeated `if-else` branching throughout the
codebase.

### Cached Singleton (`lru_cache`)

Both the Groq client (`get_llm()`) and the HuggingFace embedding model
(`_get_embedding_model()`) are relatively expensive to initialize but
remain stateless after creation.

Using `functools.lru_cache` ensures each resource is created only once
per process and reused throughout the application. This approach was
chosen instead of `st.cache_resource` so these modules remain
independent of Streamlit and fully testable with plain `pytest`.

### Centralized Error Translation

The `_translate_groq_exception()` function in `llm.py` maps the Groq
SDK's exception hierarchy into the project's
`LLMGenerationError`.

Both `safe_invoke()` and `safe_batch()` reuse this function, ensuring
consistent error messages regardless of whether requests are executed
individually or concurrently.

## 5. Prompt Engineering — Full Reasoning

All prompt templates are defined in `src/summarization/prompts.py`. This
ensures that prompt logic remains centralized and easy to maintain.

Two reusable prompt blocks are shared across every **final-output**
prompt (Stuff, Reduce, and Refine Update), so formatting and factuality
rules are defined only once.

### Shared Prompt Components

#### `_STRUCTURED_SUMMARY_FORMAT`

Defines the six-section Markdown structure used throughout the project:

- Title
- Executive Summary
- Key Points
- Important Details
- Main Takeaways
- Conclusion

The prompt also explicitly instructs the model **not** to generate any
text before or after these sections, preventing unnecessary preambles
such as *"Here's your summary:"* and keeping the UI output clean and
consistent.

---

#### `_FACTUALITY_CONSTRAINTS`

Defines rules that every final summary must follow:

- Use only the provided source content.
- Do not invent missing facts.
- If a section cannot be completed, write
  **"Not enough information in the source content."**
- Do not refer to yourself as an AI.
- Do not describe the input as a "transcript."

These constraints help minimize hallucinations while keeping summaries
grounded in the original content.

---

### STUFF_PROMPT

**Objective**

Generate the complete structured summary in a single LLM call.

**Input**

`{content}` — the entire combined document.

**Output**

A six-section Markdown summary.

**Constraints**

Uses the shared factuality rules.

**Why this design?**

Since the Stuff strategy performs only one LLM call, the prompt contains
the complete formatting instructions directly. There is no intermediate
step, so keeping the entire structure in one prompt is the simplest and
most efficient approach.

---

### MAP_PROMPT

**Objective**

Summarize a single chunk into a short intermediate summary that will
later be combined during the Reduce step.

**Input**

`{chunk_content}` — one document chunk.

**Output**

- 3–6 sentences
- Plain text only
- No Markdown
- No bullet points

**Constraints**

- Preserve important facts, names, and numbers.
- Do not introduce outside knowledge.

**Why not use the structured format?**

The Map output is never shown directly to the user. It exists only as
temporary input for the Reduce step.

Generating six Markdown sections for every chunk would waste tokens on
formatting that is discarded immediately. Keeping the output as concise
plain text improves token efficiency without affecting the final result.

**How output length is controlled**

The instruction *"3–6 sentences, no headers"* keeps each intermediate
summary compact, preventing unnecessary growth in the Reduce step's
input.

---

### REDUCE_PROMPT

**Objective**

Combine all intermediate chunk summaries into one coherent final summary.

**Input**

`{combined_summaries}` — all map outputs, labeled as `[Section N]`
in their original order.

**Output**

The standard six-section Markdown summary.

**Constraints**

Uses the shared factuality rules.

**Handling missing information**

Like the Stuff strategy, the Reduce prompt explicitly instructs the
model to write **"Not enough information in the source content."**
instead of inventing details.

This instruction is especially important because the Reduce step only
receives intermediate summaries rather than the original source text.

---

### REFINE_INITIAL_PROMPT

**Objective**

Generate the first version of the structured summary using only the
first chunk of content.

**Input**

`{first_chunk_content}`

**Output**

The standard six-section Markdown summary based on the available
information.

**Why use the full structured format?**

Unlike the Map step, this output is not temporary.

If the source contains only one chunk, this summary becomes the final
result immediately (see the `total_chunks > 1` check in
`refine_strategy.py`).

For that reason, it must already follow the complete output format.

---

### REFINE_UPDATE_PROMPT

**Objective**

Update an existing summary using newly available content from the same
source.

**Input**

- `{existing_summary}`
- `{new_content}`

**Output**

A complete six-section Markdown summary.

**Constraints**

The prompt instructs the model to:

- retain information that is still accurate
- incorporate genuinely important new information
- prefer newer information when conflicts occur
- leave the summary largely unchanged if the new content adds nothing
  significant

**Why regenerate the entire summary instead of a diff?**

Producing a complete updated summary is simpler and more reliable than
tracking incremental text differences.

This design also keeps the refinement loop straightforward—`refine_strategy.py`
only needs to maintain a single evolving summary rather than managing a
summary plus a separate changelog.

## 6. Chunking — Implementation Details

This project uses LangChain's `RecursiveCharacterTextSplitter` from
`langchain-text-splitters`.

Instead of splitting text at a fixed character limit, it attempts to
split at natural boundaries in the following order:

- Paragraph
- Sentence
- Word
- Character

A hard character-based split is only used when no better boundary is
available nearby.

### Chunk Configuration

- **`CHUNK_SIZE = 4000` characters**

  Approximately **1,000 tokens**, based on the project's
  4-characters-per-token estimation. This estimate is implemented once
  in `estimate_token_count()` (`chunker.py`) and reused throughout the
  project, including the Stuff and Reduce strategy size checks.

- **`CHUNK_OVERLAP = 400` characters (10%)**

  During development, overlap behavior was verified by inspecting real
  splitter output rather than relying only on documentation. The end of
  one chunk correctly appears at the beginning of the next, preserving
  context across chunk boundaries.

### Chunk Metadata

After splitting, `chunker.py` adds additional metadata to every chunk:

- `chunk_index`
- `start_index`

These values are generated by the project itself rather than by
`RecursiveCharacterTextSplitter`.

Indices are scoped to each `source_url`, ensuring that if multiple
documents are processed together in the future, every source will
maintain its own independent chunk numbering. Although the current
application processes one document at a time, the implementation
already supports this future use case.

### Token Estimation

Token counting is intentionally implemented as an estimate rather than
an exact tokenizer calculation.

The `openai/gpt-oss-120b` tokenizer is not available as a standard
`tiktoken` encoding, so attempting an exact count would require
guessing a tokenizer approximation. Instead of presenting false
precision, the project consistently uses a documented character-based
estimation throughout the codebase.

---

## 7. Summarization Strategies — Implementation Details

### Stuff Strategy

The Stuff strategy performs a single LLM call using:

```text
STUFF_PROMPT
      │
      ▼
get_llm()
      │
      ▼
StrOutputParser()
```

The chain is executed through `safe_invoke()`.

Before making the request, `estimate_token_count()` checks whether the
combined document exceeds
`STUFF_STRATEGY_MAX_ESTIMATED_TOKENS` (100,000 tokens).

This limit was intentionally chosen below Groq's 131K context window,
leaving sufficient space for:

- the system prompt
- model response
- token estimation margin

If the estimated size exceeds the limit, a
`SummarizationError` is raised **before** any Groq API call is made,
recommending the user switch to Map-Reduce or Refine.

---

### Map-Reduce Strategy

#### Map Step

Each chunk is summarized independently using:

```text
MAP_PROMPT
      │
      ▼
get_llm()
      │
      ▼
StrOutputParser()
```

Execution is performed through:

```python
safe_batch(
    chain,
    inputs,
    max_concurrency=MAP_REDUCE_MAX_CONCURRENCY
)
```

The default concurrency is **5**.

During development, `Runnable.batch()` concurrency was verified through
real timing experiments rather than relying solely on documentation.
Testing confirmed that multiple chunk summaries execute concurrently.

Output ordering was also verified. Even when requests complete in a
different order, the returned summaries preserve the original input
order, ensuring each chunk is assigned to the correct section during
the Reduce step.

#### Reduce Step

Intermediate summaries are combined in the following format:

```text
[Section 1]
...

[Section 2]
...
```

The combined text is then processed through:

```text
REDUCE_PROMPT
      │
      ▼
get_llm()
      │
      ▼
StrOutputParser()
```

using `safe_invoke()`.

Like the Stuff strategy, the Reduce step also performs a size check on
the combined intermediate summaries before sending them to the model.

This protects against extremely large inputs, although hierarchical or
recursive reduction is not implemented in the current version.

#### Known Trade-off

`Runnable.batch()` uses `return_exceptions=False` by default.

As a result, if any individual chunk fails during the Map phase, the
entire Map-Reduce process is aborted. Partial-result recovery is not
implemented in this version.

---

### Refine Strategy

Unlike Map-Reduce, the Refine strategy executes sequentially using a
standard Python `for` loop.

It does **not** use `safe_batch()` because every iteration depends on
the previous iteration's output.

This is an inherent property of the Refine algorithm rather than an
optimization opportunity.

The process works as follows:

1. `REFINE_INITIAL_PROMPT` generates the initial summary from the first
   chunk.

2. If `total_chunks > 1`, a single
   `REFINE_UPDATE_PROMPT | get_llm() | StrOutputParser()`
   chain is created once outside the loop.

3. The same chain is reused for every remaining chunk while updating
   the current summary.

This behavior is verified by the test suite through
`test_get_llm_called_exactly_twice_regardless_of_chunk_count`, which
confirms that `get_llm()` is called only twice regardless of the number
of chunks processed.

#### Known Trade-off

The Refine strategy does not implement checkpointing.

If an `LLMGenerationError` occurs midway through processing, previously
generated intermediate summaries are not persisted. Progress exists only
in the in-memory `current_summary` variable, so the summarization must
restart from the beginning.

## 8. Error Handling Architecture

Every extraction/summarization/LLM exception type in this project
follows the same shape:

```python
class SomeError(RuntimeError):
    def __init__(self, user_message: str, *, cause: Exception | None = None):
        ...
        self.user_message = user_message
        self.cause = cause
```

`user_message` is safe to show directly in the Streamlit UI;
`cause` (and full logging via `logger.exception` / `logger.info`)
carries the technical detail for debugging, never surfaced to the
user. `app.py` catches a single tuple of these types
(`_USER_FACING_ERRORS`) and displays `.user_message` uniformly,
regardless of which layer the failure came from.

`llm.py`'s `_translate_groq_exception` deserves particular mention: it
maps the `groq` SDK's real exception hierarchy —
`GroqError → APIError → (APIConnectionError → APITimeoutError)` and
`APIError → APIStatusError → (RateLimitError, AuthenticationError,
BadRequestError, PermissionDeniedError, NotFoundError,
InternalServerError, ...)` — verified directly via
`Exception.__mro__` inspection during development, not assumed. Both
`safe_invoke` and `safe_batch` share this single function so a
Groq failure produces an identical message whether it happened during
a single call (Stuff, Reduce, Refine) or a batched one (Map).

---

## 9. Testing Architecture

- **No test requires a real API key, a real network call, or a real
  model download.** Every external boundary (`requests`,
  `youtube-transcript-api`, `ChatGroq`/`groq`, `SentenceTransformer`)
  is mocked.
- **Two distinct mocking styles, used deliberately:**
  - `pytest-mock`'s `mocker.patch(...)` with `MagicMock` for
    boundaries where only call arguments/return values matter (e.g.
    `requests.get`, `safe_batch`'s config pass-through).
  - `RunnableLambda` "fake LLMs" (not `MagicMock`) wherever a real
    LangChain LCEL chain (`prompt | llm | parser`) needed to compose
    for real — `MagicMock` doesn't implement LangChain's `Runnable`
    protocol, so `prompt | MagicMock()` would not reliably exercise
    the same code path as production. This was verified directly
    during development (a `RunnableLambda` was confirmed to receive
    the exact same `ChatPromptValue` a real `ChatGroq` would).
- **`lru_cache`-based singletons are explicitly reset between tests**
  (`_get_embedding_model.cache_clear()`, `get_llm.cache_clear()` in
  `autouse` fixtures) — without this, the first test to run in a
  session would silently "poison" every later test with its mocked
  return value.
- **Exception objects used in tests are constructed with their real,
  library-correct arguments** (e.g. `groq.RateLimitError` needs a real
  `httpx.Response`; `NoTranscriptFound` needs a `TranscriptList`-shaped
  argument) — verified via `inspect.signature()` on the installed
  packages rather than guessed, so tests raise exactly what the real
  libraries would raise.

---

## 10. Key Engineering Decisions Log

A few decisions changed mid-build, based on what verification actually
revealed — documented here rather than presented as if the final
design was obvious from the start:

1. **LangChain package choice.** Originally planned to depend on the
   full `langchain` metapackage. Verification revealed LangChain 1.x
   deprecated `load_summarize_chain` and the legacy
   Stuff/Map-Reduce/Refine chain classes into a separate
   `langchain-classic` package. Decision: depend on `langchain-core`,
   `langchain-groq`, and `langchain-text-splitters` directly, and
   build all three strategies on LCEL primitives by hand — smaller
   dependency footprint, avoids the deprecated path entirely, and
   forces (and demonstrates) real understanding of each strategy's
   internals rather than delegating to a black-box helper.
2. **`youtube-transcript-api`'s API shape.** Most available tutorials
   reference the pre-1.0 classmethod API
   (`YouTubeTranscriptApi.get_transcript(...)`). The pinned version
   (1.2.4) uses an instance-based API (`YouTubeTranscriptApi().fetch(...)`)
   with a different, more specific exception hierarchy
   (`CouldNotRetrieveTranscript` as the common base). Confirmed via
   direct installation and `inspect.signature()` before writing the
   extractor.
3. **`safe_batch` and the `llm.py` refactor.** Building Map-Reduce's
   concurrent map step required a batch-capable analog to
   `safe_invoke`. Rather than duplicating the Groq exception-handling
   logic into a second function, the exception-translation logic was
   extracted into a shared `_translate_groq_exception` helper — a
   refactor of already-shipped, already-tested code, done because
   duplicating ~40 lines of ordered `except` clauses across two
   functions would have created a real risk of the two messages
   drifting apart over time.
4. **Chunk-size and threshold values remain heuristic.** `CHUNK_SIZE`,
   `CHUNK_OVERLAP`, `DEDUP_SIMILARITY_THRESHOLD`, and
   `STUFF_STRATEGY_MAX_ESTIMATED_TOKENS` were all set from reasoned
   engineering judgment, not from tuning against a labeled evaluation
   dataset — this project does not claim otherwise anywhere in its
   documentation.

---

## 11. Known Limitations (Technical Detail)

- **YouTube transcript blocking on shared cloud IPs is a confirmed,
  observed limitation of this deployment**, not a theoretical one —
  YouTube transcript requests succeed reliably in local development
  but can fail on Streamlit Cloud because YouTube blocks/rate-limits
  transcript requests from some shared cloud IP ranges. This is
  external to the application; `youtube_extractor.py`'s
  `RequestBlocked`/`IpBlocked` handling surfaces it as a clear message
  rather than a stack trace, but cannot prevent the underlying block.
- **No recursive/hierarchical reduce.** Extremely long content with a
  very large number of chunks could, in principle, produce combined
  intermediate summaries that still exceed the single-call size
  ceiling at the reduce step. This is guarded against (a clear error
  is raised) but not solved.
- **No checkpointing in Refine.** A failure partway through a long
  sequential refine chain loses that run's progress entirely.
- **English-only YouTube transcripts.** No translation or
  multi-language transcript selection is implemented.
- **No headless browser fallback.** JavaScript-rendered websites that
  expose little server-side HTML will fail website extraction's
  minimum-text-length check.
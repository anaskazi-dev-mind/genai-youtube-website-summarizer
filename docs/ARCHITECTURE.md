# Architecture Documentation

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [Request Flow](#request-flow)
- [Component Architecture](#component-architecture)
- [Folder Structure](#folder-structure)
- [Data Flow](#data-flow)
- [Extraction Pipeline](#extraction-pipeline)
- [Cleaning Pipeline](#cleaning-pipeline)
- [Chunking Pipeline](#chunking-pipeline)
- [Deduplication Pipeline](#deduplication-pipeline)
- [Summarization Pipeline](#summarization-pipeline)
- [Strategy Pattern Implementation](#strategy-pattern-implementation)
- [Stuff Strategy Deep Dive](#stuff-strategy-deep-dive)
- [Map-Reduce Strategy Deep Dive](#map-reduce-strategy-deep-dive)
- [Refine Strategy Deep Dive](#refine-strategy-deep-dive)
- [LLM Wrapper Architecture](#llm-wrapper-architecture)
- [Retry and Exponential Backoff](#retry-and-exponential-backoff)
- [Error Handling Architecture](#error-handling-architecture)
- [Configuration Architecture](#configuration-architecture)
- [Metadata Flow](#metadata-flow)
- [Testing Architecture](#testing-architecture)
- [Design Decisions](#design-decisions)
- [Extension Points](#extension-points)

---

## High-Level Overview
┌─────────────────────────────────────────────────────────┐ │ Streamlit Web UI │ │ (app.py) │ │ - URL Input - Strategy Selection - Result Display │ └────────────────────┬────────────────────────────────────┘ │ ▼ ┌────────────────────────────┐ │ Input Validation Layer │ │ (src/validators.py) │ │ - URL Parsing │ │ - Source Detection │ └────────┬───────────────────┘ │ ┌────────▼───────────────────┐ │ Extraction Layer │ │ (src/extractors/) │ │ - YouTube Transcripts │ │ - Website Content │ └────────┬───────────────────┘ │ ┌────────▼───────────────────┐ │ Processing Pipeline │ │ (src/processing/) │ │ 1. Cleaning │ │ 2. Chunking │ │ 3. Deduplication │ └────────┬───────────────────┘ │ ┌────────▼───────────────────┐ │ Summarization Pipeline │ │ (src/summarization/) │ │ - Stuff │ │ - Map-Reduce │ │ - Refine │ └────────┬───────────────────┘ │ ▼ ┌──────────────────┐ │ Markdown Summary │ └──────────────────┘



---

## Request Flow

User Input URL │ ├─→ Is URL valid? (validators.py) │ └─→ ❌ Invalid → User Error Message │ ├─→ Extract source type (YouTube vs Website) │ ├─→ Fetch & extract content │ ├─→ YouTube: Fetch transcript API │ └─→ Website: HTTP request → trafilatura/BeautifulSoup │ └─→ ❌ Extract fails → User Error Message │ ├─→ Create LangChain Document │ ├─→ Clean text (normalize, remove noise) │ ├─→ Split into chunks (RecursiveCharacterTextSplitter) │ └─→ ❌ Too many chunks (>15) → User Error Message │ ├─→ Deduplicate chunks (semantic similarity) │ └─→ ❌ All chunks removed → User Error Message │ ├─→ User selects strategy (Stuff/Map-Reduce/Refine) │ ├─→ Strategy.summarize(chunks) │ ├─→ Stuff: Single LLM call │ ├─→ Map-Reduce: N concurrent calls + 1 reduce call │ ├─→ Refine: 1 initial + N-1 sequential calls │ └─→ ❌ Groq error → Retry with exponential backoff → User Error Message │ └─→ Return SummaryResult (Markdown)


---

## Component Architecture

### Layer 1: Validation & Routing (src/validators.py)

**Responsibility**: Parse and classify input

**Exports**:
- `SourceType` enum: YOUTUBE, WEBSITE
- `URLValidationError` exception
- `is_valid_url(url)` → bool
- `detect_source_type(url)` → SourceType
- `extract_youtube_video_id(url)` → str (11-char ID)

**Key Design**:
- Pure functions with no side effects
- No network calls (fast, testable)
- Exhaustive YouTube URL format support

### Layer 2: Extraction (src/extractors/)

**Responsibility**: Convert source → LangChain Document

**youtube_extractor.py**:
- `fetch_youtube_document(url)` → Document
- Handles all YouTube API failures gracefully
- Metadata: video_id, language, is_generated

**website_extractor.py**:
- `fetch_website_document(url)` → Document
- Two-tier extraction: trafilatura → BeautifulSoup fallback
- Metadata: title (HTML-escaped), source_url

### Layer 3: Processing (src/processing/)

**cleaner.py**:
- `clean_document(doc)` → Document
- Unicode normalization, whitespace collapse, annotation removal
- Returns new Document (non-mutating)

**chunker.py**:
- `split_documents(docs, enforce_max_chunks=True)` → List[Document]
- RecursiveCharacterTextSplitter with configurable size/overlap
- Adds chunk_index and start_index metadata

**deduplicator.py**:
- `deduplicate_chunks(chunks)` → List[Document]
- Loads HuggingFace embedding model once (cached)
- Greedy O(n²) comparison against kept chunks

### Layer 4: Summarization (src/summarization/)

**base_strategy.py**:
- `SummarizationStrategy` abstract base class
- `SummaryResult` dataclass (content, strategy, chunk_count, source_metadata)
- `_validate_documents()` and `_source_metadata_from()` helpers

**llm.py**:
- `get_llm()` → ChatGroq singleton
- `safe_invoke(chain, input)` → output (with retry)
- `safe_batch(chain, inputs, max_concurrency=5)` → List[output] (with retry)
- Exponential backoff on RateLimitError

**prompts.py**:
- All prompt templates centralized
- Shared structured output format
- Strategy-specific customizations only

**Concrete Strategies**:
- `StuffStrategy`: Single LLM call
- `MapReduceStrategy`: Concurrent map + single reduce
- `RefineStrategy`: Sequential refinement

**strategy_factory.py**:
- `available_strategies()` → List[str]
- `get_strategy(name)` → SummarizationStrategy instance

### Layer 5: UI (app.py)

**Responsibility**: Streamlit frontend only

**Key Patterns**:
- Session state persistence
- Status container for pipeline progress
- Error handling with user-safe messages
- No business logic (delegated to src/)

---

## Folder Structure

src/ ├── init.py ├── config.py # Configuration constants (all in one place) ├── logger.py # Logging setup ├── validators.py # URL validation & source detection │ ├── extractors/ │ ├── init.py │ ├── youtube_extractor.py │ └── website_extractor.py │ ├── processing/ │ ├── init.py │ ├── cleaner.py # Text cleaning │ ├── chunker.py # Text splitting │ └── deduplicator.py # Semantic deduplication │ └── summarization/ ├── init.py ├── base_strategy.py # Abstract interface ├── llm.py # Groq wrapper + retry logic ├── prompts.py # Prompt templates ├── stuff_strategy.py # Single-call strategy ├── map_reduce_strategy.py # Parallel + reduce strategy ├── refine_strategy.py # Sequential refinement strategy └── strategy_factory.py # Strategy registry & selection

tests/ ├── test_validators.py ├── test_youtube_extractor.py ├── test_website_extractor.py ├── test_cleaner.py ├── test_chunker.py ├── test_deduplicator.py ├── test_llm.py ├── test_stuff_strategy.py ├── test_map_reduce_strategy.py ├── test_refine_strategy.py └── test_strategy_factory.py


---

## Data Flow

### Document Object

All components pass `langchain_core.documents.Document` objects:

```
Document(
    page_content: str,  # The actual text
    metadata: {
        "source_type": "youtube" | "website",
        "source_url": str,
        "video_id": str,  # YouTube only
        "language": str,  # YouTube only
        "is_generated": bool,  # YouTube only
        "title": str,  # Website only
        
        # Added by chunker:
        "chunk_index": int,
        "start_index": int,
    }
)
Metadata Preservation
Each component adds information without modifying source metadata:


Extraction:
  video_id, language, is_generated (YouTube)
  title (Website)

Cleaning:
  (metadata unchanged)

Chunking:
  chunk_index (per source)
  start_index

Deduplication:
  (metadata unchanged, only list filtered)

Summarization:
  Filters metadata to displayable subset
Extraction Pipeline
YouTube Extraction

URL: "https://www.youtube.com/watch?v=xAt1xcC6qfM"
    │
    ├─→ extract_youtube_video_id(url)
    │   └─→ "xAt1xcC6qfM"
    │
    ├─→ YouTubeTranscriptApi().fetch(video_id, languages=("en", "en-US", "en-GB"))
    │   └─→ List of {"text": "...", "start": ..., "duration": ...} objects
    │
    ├─→ Join all text snippets
    │   └─→ "what's the biggest misunderstanding about you..."
    │
    └─→ Document(
        page_content="...",
        metadata={
            "source_type": "youtube",
            "video_id": "xAt1xcC6qfM",
            "source_url": url,
            "language": "en",
            "is_generated": True/False
        }
    )
Error Scenarios (all caught and converted to YouTubeExtractionError):

TranscriptsDisabled: Uploader disabled captions
NoTranscriptFound: No English transcript
AgeRestricted: Login required
VideoUnplayable/InvalidVideoId: Video not found
RequestBlocked/IpBlocked: YouTube blocking requests
CouldNotRetrieveTranscript: Generic failure
Empty transcript: Transcript exists but has no text
Website Extraction

URL: "https://example.com/article"
    │
    ├─→ requests.get(url, headers={"User-Agent": "..."}, timeout=15)
    │   └─→ HTML response
    │
    ├─→ trafilatura.extract(html, output_format="json", with_metadata=True)
    │   └─→ JSON: {"text": "...", "title": "...", ...} or None
    │
    ├─→ Parse JSON, check if text >= MIN_ACCEPTABLE_TEXT_LENGTH (200 chars)
    │   │
    │   └─→ If too short OR None → BeautifulSoup fallback
    │       ├─→ Remove <script>, <style>, <nav>, <header>, <footer>
    │       ├─→ Extract all <p> tags
    │       └─→ Re-check min length
    │
    └─→ Document(
        page_content="Artificial intelligence is...",
        metadata={
            "source_type": "website",
            "source_url": url,
            "title": "Artificial Intelligence - Wikipedia"
        }
    )
Error Scenarios:

Timeout: Website took >15 seconds
ConnectionError: Network unreachable
HTTPError: 404, 403, 500, etc.
Non-HTML: Content-Type not HTML
Too little content: <200 chars after extraction
Cleaning Pipeline

def clean_document(doc: Document) -> Document:
    text = doc.page_content
    
    # 1. Unicode normalization (NFKC)
    #    Example: "é" (e + combining accent) → "é" (single char)
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Replace non-breaking spaces with regular spaces
    text = text.replace("\xa0", " ")
    
    # 3. Remove bracketed YouTube annotations
    #    Remove: [Music], [Applause], [inaudible], etc.
    text = re.sub(r"\[(music|applause|...)\]", " ", text, flags=re.IGNORECASE)
    
    # 4. Collapse excess whitespace
    #    "hello  \n\n  world" → "hello world"
    text = re.sub(r"\s{2,}", " ", text)
    
    # 5. Strip leading/trailing whitespace
    text = text.strip()
    
    return Document(page_content=text, metadata=doc.metadata)
Design Philosophy: Conservative only

No sentence restructuring
No stopword removal
No paraphrasing
Preserves meaning while removing noise
Chunking Pipeline

def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,           # 6000 chars
        chunk_overlap=CHUNK_OVERLAP,     # 500 chars (10%)
        length_function=len,
        add_start_index=True
    )
    
    # Split preserving boundaries in order of:
    # 1. Paragraph breaks (\n\n)
    # 2. Sentence endings (. ! ?)
    # 3. Word spaces
    # 4. Character cut (worst case)
    chunks = splitter.split_documents(documents)
    
    # Safety: Fail if too many chunks (cost control)
    if len(chunks) > MAX_CHUNKS_PER_CONTENT (15):
        raise SummarizationError("Content too long...")
    
    # Add chunk_index metadata (per-source)
    for chunk in chunks:
        source_key = chunk.metadata["source_url"]
        chunk.metadata["chunk_index"] = counter[source_key]
        counter[source_key] += 1
    
    return chunks
Example:


Input: 186,000 character Wikipedia article
CHUNK_SIZE=6000, CHUNK_OVERLAP=500

Output:
Chunk 0: chars 0-6000 (includes "start_index": 0, "chunk_index": 0)
Chunk 1: chars 5500-11500 (includes "start_index": 5500, "chunk_index": 1)
Chunk 2: chars 11000-17000 (includes "start_index": 11000, "chunk_index": 2)
... 30+ more chunks ...

If >15 chunks → Error: "Content too long to process"
Deduplication Pipeline

def deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    # 1. Load embedding model (cached)
    model = _get_embedding_model()  # "sentence-transformers/all-MiniLM-L6-v2"
    
    # 2. Encode all chunks to vectors
    embeddings = model.encode(
        [chunk.page_content for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True  # Cosine similarity
    )
    
    # 3. Greedy comparison (keep first, drop duplicates)
    kept_chunks = []
    kept_embeddings = []
    
    for chunk, embedding in zip(chunks, embeddings):
        # Compare against already-kept chunks
        if kept_embeddings:
            similarities = util.cos_sim(embedding, np.vstack(kept_embeddings))[0]
            max_similarity = similarities.max()
        else:
            max_similarity = 0.0
        
        # If >92% similar to any kept chunk, drop it
        if max_similarity >= DEDUP_SIMILARITY_THRESHOLD (0.92):
            logger.info(f"Dropping chunk (similarity={max_similarity:.3f})")
            continue
        
        # Otherwise keep it
        kept_chunks.append(chunk)
        kept_embeddings.append(embedding)
    
    return kept_chunks
Example:


Input: 7 chunks from a YouTube video

Chunk 0: "Introduction to AI..."
  └─→ Keep (first)

Chunk 1: "Introduction to AI..." (repeated intro)
  └─→ Similarity: 0.95 → Drop

Chunk 2: "Key concepts of AI..."
  └─→ Similarity to Chunk 0: 0.72, Chunk 1: N/A
  └─→ Keep (not duplicate)

...continue...

Output: 6 chunks (1 duplicate removed)
Summarization Pipeline
Entry Point

strategy = get_strategy("map_reduce")  # or "stuff" or "refine"
result = strategy.summarize(chunks)    # → SummaryResult
SummaryResult

@dataclass(frozen=True)
class SummaryResult:
    content: str           # Full Markdown summary
    strategy: str          # "stuff" | "map_reduce" | "refine"
    chunk_count: int       # How many chunks processed
    source_metadata: dict  # {source_type, source_url, title/video_id, ...}
Strategy Pattern Implementation

                SummarizationStrategy (Abstract)
                         △
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    StuffStrategy  MapReduceStrategy  RefineStrategy
Interface:


class SummarizationStrategy(ABC):
    name: str  # "stuff" | "map_reduce" | "refine"
    
    @abstractmethod
    def summarize(self, documents: List[Document]) -> SummaryResult:
        ...
    
    @staticmethod
    def _validate_documents(documents):
        # Raises if empty
        
    @staticmethod
    def _source_metadata_from(documents):
        # Extracts displayable fields from first doc
Selection:


# Registry in strategy_factory.py
_STRATEGY_REGISTRY = {
    "stuff": StuffStrategy,
    "map_reduce": MapReduceStrategy,
    "refine": RefineStrategy,
}

# Factory function
def get_strategy(name: str) -> SummarizationStrategy:
    return _STRATEGY_REGISTRY[name]()
Stuff Strategy Deep Dive
Use Case: Short to medium content that fits in one LLM call

Flow:

Input: List[Document] (chunks)
    │
    ├─→ Validate (not empty)
    │
    ├─→ Combine all chunks into single text
    │   Example: "chunk1_text\n\nchunk2_text\n\nchunk3_text"
    │
    ├─→ Estimate tokens: len(combined) / 4
    │   └─→ If >100K tokens → Error: Use Map-Reduce/Refine
    │
    ├─→ Log: "Summarizing N chunks in 1 call"
    │
    ├─→ Build chain:
    │   STUFF_PROMPT | get_llm() | StrOutputParser()
    │   
    │   STUFF_PROMPT inputs:
    │   - System message (summarizer role + constraints)
    │   - Human message: {content}
    │
    ├─→ safe_invoke(chain, {"content": combined_text})
    │   └─→ Returns: Full Markdown summary
    │
    └─→ Return SummaryResult
        - content: summary
        - strategy: "stuff"
        - chunk_count: len(chunks)
        - source_metadata: {...}
Prompt Template:

SYSTEM:
"You are an expert content summarizer. You will be given the full text of
either a YouTube video transcript or a website article. Produce a structured
summary of it.

Constraints:
- Base the summary ONLY on the provided content...
- Preserve factual accuracy...

Format your response using EXACTLY this structure, in Markdown:
## Title
... (title, max 12 words)
## Executive Summary
... (2-3 sentences)
## Key Points
- (4-6 bullets)
... (more sections)
"

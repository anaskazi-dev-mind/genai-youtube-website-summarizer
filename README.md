# 📝 AI YouTube & Website Summarizer

An LLM-powered summarization application that transforms YouTube videos and website articles into structured, actionable summaries using three interchangeable strategies—Stuff, Map-Reduce, and Refine. Built with LangChain, powered by Groq, and deployed on Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-135%20passing-brightgreen)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Sources](#supported-sources)
- [Summarization Strategies](#summarization-strategies)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
- [Running Tests](#running-tests)
- [Configuration Options](#configuration-options)
- [Error Handling](#error-handling)
- [Logging](#logging)
- [Rate Limit Handling](#rate-limit-handling)
- [Example Usage](#example-usage)
- [Limitations](#limitations)
- [Performance Considerations](#performance-considerations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Architecture](#architecture)

---

## Overview

This project solves the problem of consuming large volumes of long-form video and article content. Instead of watching 1-hour videos or reading 10,000-word articles, users can paste a URL and get a concise, well-structured summary in seconds.

The application extracts content, cleans it, splits it into manageable chunks, removes redundant content using semantic similarity, and generates summaries using one of three LLM-powered strategies, each optimized for different content lengths and use cases.

---

## Features

✅ **Multi-source summarization**
- YouTube video transcripts (English-language only)
- Website articles and blog posts
- Both sources processed through a unified pipeline

✅ **Three summarization strategies**
- **Stuff**: Fast, single-call summarization for short content
- **Map-Reduce**: Parallel chunk processing for long documents
- **Refine**: Sequential updates preserving narrative continuity

✅ **Intelligent content processing**
- Automatic text cleaning (normalization, noise removal)
- Smart chunking with sentence/paragraph boundary preservation
- Semantic deduplication using HuggingFace embeddings
- Live pipeline progress tracking in the UI

✅ **Structured output**
- Title, Executive Summary, Key Points, Important Details, Takeaways, Conclusion
- Markdown format for easy sharing and rendering
- Consistent structure across all strategies

✅ **Production-ready quality**
- 135 automated tests with >95% code coverage
- Comprehensive error handling for all failure scenarios
- Security: No hardcoded API keys, environment-based secrets only
- Logging: Detailed pipeline telemetry without leaking sensitive data
- Rate limit resilience: Exponential backoff retry on Groq quota exhaustion

✅ **Developer-friendly**
- Modular architecture: Easy to extend with new extractors or strategies
- Single responsibility per component
- Centralized configuration in `src/config.py`
- Clean separation of UI (app.py) from business logic (src/)

---

## Supported Sources

### YouTube Videos
- **Format**: URLs like `youtube.com/watch?v=VIDEO_ID`, `youtu.be/VIDEO_ID`, `/shorts/VIDEO_ID`, `/live/VIDEO_ID`, `/embed/VIDEO_ID`
- **Language**: English transcripts only
- **Limitations**: 
  - Requires YouTube captions/transcripts to be available
  - May fail on Streamlit Cloud due to YouTube blocking shared cloud IPs
  - Works reliably in local development

### Websites & Articles
- **Format**: Any `http://` or `https://` URL
- **Method**: Smart content extraction using trafilatura (with BeautifulSoup fallback)
- **Works best**: Standard blog posts, news articles, documentation
- **Limitations**:
  - No headless browser: JavaScript-rendered content may not extract properly
  - Paywalled/anti-bot content may fail extraction
  - Minimum 200 characters of extractable content required

---

## Summarization Strategies

| Strategy | How It Works | Best For | Latency | API Calls | Cost |
|----------|-------------|----------|---------|-----------|------|
| **Stuff** | Combines all chunks into one prompt, single LLM call | Short to medium content (<6000 chars) | ⚡ Fastest | 1 | Lowest |
| **Map-Reduce** | Summarizes each chunk independently (concurrent), combines in reduce step | Long content with natural section breaks | 🔄 Medium | 1 + N | Medium |
| **Refine** | Initial summary, then sequentially refines with each chunk | Long narrative content, needs context preservation | 🐢 Slowest | 1 + N | Medium |

**Selection Guide:**
- Want speed? Use **Stuff** for articles/videos under 15 minutes
- Processing long content? Use **Map-Reduce** for parallel efficiency
- Summarizing novels/podcasts? Use **Refine** to preserve story arc

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM Inference** | Groq (`openai/gpt-oss-120b`) | Fast, 131K context window |
| **LLM Framework** | LangChain 1.x modular packages | Document/prompt/chain orchestration |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` | Semantic deduplication only (~90MB, local) |
| **UI/Frontend** | Streamlit | Zero-configuration web interface |
| **YouTube Extraction** | `youtube-transcript-api` | Transcript fetching |
| **Web Extraction** | `trafilatura` + `beautifulsoup4` | Content extraction |
| **Testing** | Pytest + pytest-mock | Unit/integration testing |

**Why these choices?**
- **Groq**: Fastest LLM inference; free tier good for experimentation
- **LangChain modular**: Lightweight, no dependency on deprecated `langchain` monolith
- **Sentence Transformers**: Tiny, no GPU needed, works on Streamlit Cloud free tier
- **Streamlit**: Fastest way to build a data/ML app without frontend expertise
- **Trafilatura**: Purpose-built for main content extraction from web pages
- **Pytest**: Industry standard Python testing with excellent mocking support

---

## Project Structure

genai-youtube-website-summarizer/ ├── src/ │ ├── init.py │ ├── config.py # Centralized configuration (models, API keys, timeouts) │ ├── logger.py # Logging setup │ ├── validators.py # URL validation & source detection │ │ │ ├── extractors/ # Extract content from sources │ │ ├── youtube_extractor.py │ │ └── website_extractor.py │ │ │ ├── processing/ # Clean, chunk, deduplicate │ │ ├── cleaner.py # Text normalization │ │ ├── chunker.py # LangChain RecursiveCharacterTextSplitter wrapper │ │ └── deduplicator.py # Semantic deduplication via embeddings │ │ │ └── summarization/ # Summarization strategies & LLM wrapper │ ├── base_strategy.py # Abstract base class & SummaryResult dataclass │ ├── llm.py # Groq wrapper with exponential backoff retry │ ├── prompts.py # Centralized prompt templates │ ├── stuff_strategy.py # Single-call summarization │ ├── map_reduce_strategy.py # Parallel + reduce summarization │ ├── refine_strategy.py # Sequential refinement summarization │ └── strategy_factory.py # Strategy selection & registration │ ├── tests/ # 135 automated tests │ ├── test_validators.py │ ├── test_youtube_extractor.py │ ├── test_website_extractor.py │ ├── test_cleaner.py │ ├── test_chunker.py │ ├── test_deduplicator.py │ ├── test_llm.py │ ├── test_stuff_strategy.py │ ├── test_map_reduce_strategy.py │ ├── test_refine_strategy.py │ ├── test_strategy_factory.py │ └── (more tests...) │ ├── app.py # Streamlit UI entry point ├── requirements.txt # Python dependencies ├── .env.example # Configuration template ├── .gitignore # Git ignore rules ├── pytest.ini # Pytest configuration └── README.md # This file

Code

---

## Installation

### Prerequisites
- Python 3.9+
- Groq API key (free tier at https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/anaskazi-dev-mind/genai-youtube-website-summarizer.git
cd genai-youtube-website-summarizer
2. Create a virtual environment
bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
# or
venv\Scripts\activate          # Windows
3. Install dependencies
bash
pip install -r requirements.txt
4. Set up environment variables
bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
5. Run the application
bash
streamlit run app.py
The application will open at http://localhost:8501.

Environment Variables
Required and optional configuration:

Variable	Required	Default	Description
GROQ_API_KEY	✅ Yes	N/A	API key from https://console.groq.com
GROQ_MODEL_NAME	No	openai/gpt-oss-120b	Override LLM model
LOG_LEVEL	No	INFO	Logging verbosity (DEBUG, INFO, WARNING, ERROR)
GROQ_RATE_LIMIT_RETRY_ATTEMPTS	No	5	Retry attempts on rate limit (1-10 recommended)
GROQ_RATE_LIMIT_RETRY_BASE_DELAY	No	1	Initial backoff delay in seconds (exponential: 1s, 2s, 4s, ...)
Example .env:

bash
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL_NAME=openai/gpt-oss-120b
LOG_LEVEL=INFO
GROQ_RATE_LIMIT_RETRY_ATTEMPTS=5
GROQ_RATE_LIMIT_RETRY_BASE_DELAY=1
Running Locally
Start the application
bash
streamlit run app.py
Basic usage
Paste a YouTube video URL or website URL
Select a summarization strategy (Stuff, Map-Reduce, or Refine)
Click "Summarize"
View the structured summary and metadata
Advanced: Direct Python usage (non-Streamlit)

from src.validators import detect_source_type, SourceType
from src.extractors.youtube_extractor import fetch_youtube_document
from src.extractors.website_extractor import fetch_website_document
from src.processing.cleaner import clean_document
from src.processing.chunker import split_documents
from src.processing.deduplicator import deduplicate_chunks
from src.summarization.strategy_factory import get_strategy

# Extract
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
source = detect_source_type(url)
doc = fetch_youtube_document(url) if source == SourceType.YOUTUBE else fetch_website_document(url)

# Process
doc = clean_document(doc)
chunks = split_documents([doc])
chunks = deduplicate_chunks(chunks)

# Summarize
strategy = get_strategy("map_reduce")  # or "stuff" or "refine"
result = strategy.summarize(chunks)

print(result.content)  # Print Markdown summary
Running Tests
Run all tests
bash
pytest -v
Run specific test file
bash
pytest tests/test_chunker.py -v
Run with coverage report
bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View coverage
Test categories
URL Validation: tests/test_validators.py
Content Extraction: tests/test_youtube_extractor.py, tests/test_website_extractor.py
Text Processing: tests/test_cleaner.py, tests/test_chunker.py
Deduplication: tests/test_deduplicator.py
LLM & Retry Logic: tests/test_llm.py
Strategies: tests/test_stuff_strategy.py, tests/test_map_reduce_strategy.py, tests/test_refine_strategy.py
All tests use mocked external APIs (Groq, YouTube, HuggingFace) and run without network access or API keys.

Configuration Options
All configuration is centralized in src/config.py and src/config.py. Modify via environment variables or edit the file directly.

Chunking Configuration

CHUNK_SIZE = 6000              # Characters per chunk (~1,500 tokens)
CHUNK_OVERLAP = 500            # Overlap between chunks (10%)
MAX_CHUNKS_PER_CONTENT = 15    # Safety limit (cost control)
Impact: Larger chunks = fewer API calls but more context per request. Smaller chunks = more calls but lower per-request token count.

LLM Configuration

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"  # Main model
GROQ_TEMPERATURE = 0.2                      # Low temp = more factual
GROQ_REQUEST_TIMEOUT_SECONDS = 120          # 2-minute timeout
Deduplication Configuration

HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEDUP_SIMILARITY_THRESHOLD = 0.92           # Remove >92% similar chunks
HUGGINGFACE_EMBEDDING_TIMEOUT_SECONDS = 30  # Model load timeout
Summarization Limits

STUFF_STRATEGY_MAX_ESTIMATED_TOKENS = 100_000  # Max single-call size
MAP_REDUCE_MAX_CONCURRENCY = 1                  # Concurrent map tasks
Error Handling
The application handles all common failure scenarios with clear, user-friendly messages:

URL/Validation Errors
Invalid URLs
Unsupported URL formats
YouTube-Specific Errors
Video not found or unavailable
Transcripts disabled by uploader
No English transcript available
Age-restricted videos
YouTube blocking transcript requests (Streamlit Cloud issue)
Website-Specific Errors
Connection timeouts
HTTP errors (404, 500, etc.)
Non-HTML responses
Insufficient extractable content
Processing Errors
Content too long (exceeds chunk limit)
All chunks removed by deduplication
HuggingFace model download timeout
LLM/Groq Errors
Authentication failures
Rate limit exceeded
Request timeouts
Oversized inputs for strategy
All errors are logged with full technical details while displaying sanitized messages to users.

Logging
Logging is configured in src/logger.py with per-module granularity.

Log Levels
DEBUG: Verbose internal details (token counts, chunk indices, etc.)
INFO: Pipeline milestones (extraction, chunking, summarization steps)
WARNING: Rate limits, slow operations, non-critical issues
ERROR: Failures and exceptions (with technical details)
Enable debug logging
bash
LOG_LEVEL=DEBUG streamlit run app.py
Log output example
Code
2026-08-15 17:26:36 | INFO     | src.extractors.youtube_extractor | Extracted YouTube transcript (33750 chars, language=en)
2026-08-15 17:26:36 | INFO     | src.processing.chunker | Split 1 document(s) into 7 chunks
2026-08-15 17:26:45 | INFO     | src.processing.deduplicator | Deduplication: 7 chunks -> 6 chunks (1 dropped, threshold=0.92)
2026-08-15 17:26:50 | WARNING  | src.summarization.llm | Rate limit hit (attempt 1/5). Retrying in 1 seconds...
2026-08-15 17:26:51 | INFO     | src.summarization.refine_strategy | Refine: updating summary with chunk 2 of 7
Logs never contain API keys or sensitive data by design.

Rate Limit Handling
The Groq free tier has rate limits (~30 requests/minute). This application includes automatic recovery:

Exponential Backoff Retry
When a rate limit is hit:

Wait 1 second, retry
If fails: wait 2 seconds, retry
If fails: wait 4 seconds, retry
If fails: wait 8 seconds, retry
If fails: wait 16 seconds, retry
After 5 attempts, fail with user message
Configuration:

bash
GROQ_RATE_LIMIT_RETRY_ATTEMPTS=5        # Total attempts
GROQ_RATE_LIMIT_RETRY_BASE_DELAY=1      # Initial backoff (seconds)
Example: A rate-limited Map-Reduce with 3 chunks will:

Attempt 1: Fail immediately (rate limit)
Wait 1s, Attempt 2: Retry all 3 chunks (concurrent)
If still rate-limited: Wait 2s, Attempt 3
Continue with exponential backoff up to 5 attempts
Cost of retries: Zero additional cost—retries consume no extra quota since they fail at the same second.

User guidance on rate limits
Use Stuff strategy for faster completion (1 API call vs. N+1)
Try shorter content (fewer chunks = fewer calls)
Wait 2-5 minutes for quotas to reset
Consider longer GROQ_RATE_LIMIT_RETRY_ATTEMPTS for free tier
Example Usage
Summarize a YouTube Video
Code
URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Strategy: Map-Reduce
Result: A structured summary with title, key points, and takeaways
Summarize a Blog Post
Code
URL: https://example.com/article/best-python-practices
Strategy: Stuff (if article is short) or Map-Reduce (if long)
Result: Executive summary with important details extracted
Summarize a Long Article Series
Code
URL: https://example.com/series-part-1
Strategy: Refine (preserves narrative continuity across sections)
Result: Summary that respects the original article's structure
Limitations
⚠️ Content Source Limitations

YouTube transcripts must be in English or auto-generated
YouTube may block transcript requests from Streamlit Cloud
Websites with JavaScript rendering may fail extraction
Paywalled or anti-bot-protected content may be inaccessible
⚠️ Summarization Limitations

Stuff strategy fails on content >100K tokens
Map-Reduce Reduce step may fail on >100K combined intermediate summaries (no recursive reduce)
Refine strategy has no checkpointing (restart required if it fails mid-processing)
Chunk size/overlap/dedup threshold are practical defaults, not empirically tuned
⚠️ API Limitations

Groq free tier: ~30 requests/minute (quota resets every minute)
No batch processing: Each chunk request counts as one API call
Model selection is limited to Groq's currently available models
⚠️ Operational Limitations

No multi-user session management
No result caching across requests
Temporary in-memory storage only (no persistence)
Performance Considerations
Latency (Time to Summary)
Content Type	Length	Stuff	Map-Reduce	Refine
YouTube Short	<1 min	<5s	~10s	~15s
YouTube Video	10-20 min	Fails (too long)	30-60s	60-90s
Blog Post	2,000-5,000 words	10-20s	20-30s	30-40s
Long Article	10,000+ words	Fails	60-120s	120-180s
Notes:

Stuff is fastest but limited to short content
Map-Reduce parallelizes chunk processing (faster for long content)
Refine is sequential but preserves narrative flow
Rate limits may add 1-5 minute wait time on free Groq tier
Cost (API Calls)
Strategy	Calls	Cost on Free Tier
Stuff (1 chunk)	1	✅ Free
Map-Reduce (5 chunks)	6 (5 map + 1 reduce)	~$0.01-0.05 (paid)
Refine (5 chunks)	5 (1 initial + 4 updates)	~$0.01-0.04 (paid)
Free tier: Unlimited calls, but rate-limited to ~30/minute.

Memory Usage
HuggingFace model: ~90MB (cached locally)
Document storage: ~1MB per 1000 chunks
No streaming: Full content loaded before processing
Future Improvements
🎯 Short-term

Multilingual YouTube transcript support with auto-translation
Streaming chunk processing (reduce memory footprint)
Result caching with TTL
🎯 Medium-term

Hierarchical/recursive reduction for 100K+ token summaries
Refine strategy checkpointing and recovery
Headless browser fallback for JS-rendered websites
Multiple LLM provider support (Claude, Gemini, etc.)
🎯 Long-term

Fast summarization mode using smaller models (gpt-oss-20b)
Vector database integration for semantic search in summaries
Multi-document summarization
Structured extraction (extract facts, figures, entities)
Fine-tuned models for domain-specific summarization
Contributing
Contributions welcome! The codebase is modular and well-tested.

To add a new feature:

Create a branch: git checkout -b feature/my-feature
Make changes with tests: pytest tests/
Ensure full test coverage: pytest --cov=src
Submit a PR with description of changes
To add a new summarization strategy:

Inherit from SummarizationStrategy in src/summarization/base_strategy.py
Implement the summarize() method
Register in src/summarization/strategy_factory.py
Add tests in tests/test_my_strategy.py
Update UI in app.py to include new strategy option
License
This project is open source. See LICENSE file for details.

Quick Links
Live Demo: https://genai-youtube-website-summarizer-ezjvepfm9fc7dzy9dbtbdo.streamlit.app/
Groq Console: https://console.groq.com
LangChain Docs: https://python.langchain.com
Architecture Details: See ARCHITECTURE.md
Support
Issues: https://github.com/anaskazi-dev-mind/genai-youtube-website-summarizer/issues
Discussions: https://github.com/anaskazi-dev-mind/genai-youtube-website-summarizer/discussions
Built with ❤️ using LangChain, Groq, and Streamlit
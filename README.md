# 📝 AI YouTube & Website Summarizer

An LLM-powered summarization application that transforms YouTube videos
and website articles into clear, structured summaries using three
interchangeable summarization strategies—Stuff, Map-Reduce, and Refine—
built with LangChain and powered by Groq.

> **Status:** The core pipeline and Streamlit UI are complete and deployed on Streamlit Cloud. Website summarization works end-to-end using real Groq API calls. YouTube summarization works reliably in local development. On Streamlit Cloud, transcript extraction may fail because YouTube blocks transcript requests from many shared cloud IP addresses—a platform limitation rather than an application issue.

---

## Table of Contents

- [Problem & Solution](#problem--solution)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Model Selection](#model-selection)
- [Summarization Strategies](#summarization-strategies)
- [YouTube Pipeline](#youtube-pipeline)
- [Website Pipeline](#website-pipeline)
- [Chunking Strategy](#chunking-strategy)
- [Prompt Engineering](#prompt-engineering)
- [Role of Each Core Technology](#role-of-each-core-technology)
- [Error Handling](#error-handling)
- [Security](#security)
- [Testing](#testing)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [GitHub & Deployment Workflow](#github--deployment-workflow)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Screenshots](#screenshots)
- [Live Demo](#live-demo)

---

## Problem & Solution

**Problem:** Long-form YouTube videos and website articles take
significant time to consume. Many existing summarization tools focus on
a single content source or rely on a single summarization approach,
making them less effective across different types and lengths of
content.

**Solution:** This project extracts content from a YouTube video or a
website URL, cleans and chunks the text, optionally removes
near-duplicate content, and generates a structured Markdown summary
using one of three interchangeable summarization strategies—Stuff,
Map-Reduce, or Refine. Each strategy is designed for different content
lengths and structures while producing a consistent output format.

## Features

- Summarizes both YouTube videos and website/articles from a single application
- Website summarization works in both local and Streamlit Cloud deployments
- YouTube transcript summarization is fully supported locally (cloud deployments may be affected by YouTube transcript blocking on shared cloud IPs)
- Supports three summarization strategies:
  - **Stuff** – Fast, single-pass summarization for short to medium content
  - **Map-Reduce** – Best for long documents using parallel chunk summarization
  - **Refine** – Sequential summarization that preserves narrative continuity
- Generates structured Markdown summaries with:
  - Title
  - Executive Summary
  - Key Points
  - Important Details
  - Main Takeaways
  - Conclusion
- Removes near-duplicate content using semantic similarity (HuggingFace embeddings)
- Displays live pipeline progress (Extraction → Cleaning → Chunking → Deduplication → Summarization)
- Provides clear, user-friendly error messages for all supported failure scenarios
- Keeps API keys secure using environment variables (`.env`) and Streamlit Secrets
- 135 automated tests covering URL validation, content extraction, text cleaning, chunking, semantic deduplication, all summarization strategies, strategy selection, configuration, and LLM error handling

## Architecture

The application follows a modular pipeline where each stage has a single responsibility and can be tested independently.

```text
                        INPUT URL
                            │
                            ▼
        URL Validation & Source Detection
             (src/validators.py)
                            │
                            ▼
             Content Extraction
          (src/extractors/)
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
     (Stuff / Map-Reduce / Refine)
      (src/summarization/)
                            │
                            ▼
            Structured Summary
                            │
                            ▼
               Streamlit UI
                 (app.py)
```

### Design Overview

The project is organized into independent layers, making the codebase modular, maintainable, and easy to extend.

- **Validation Layer** validates the input URL and detects whether it is a YouTube video or a website.
- **Extraction Layer** contains source-specific extractors (`youtube_extractor.py` and `website_extractor.py`) that convert different inputs into a common LangChain `Document` object.
- **Processing Layer** performs text cleaning, chunking, and semantic deduplication before sending content to the LLM.
- **Summarization Layer** implements the Strategy design pattern. All summarization strategies inherit from the common `SummarizationStrategy` interface defined in `base_strategy.py` and are selected dynamically through `strategy_factory.py`.
- **Presentation Layer** (`app.py`) contains only Streamlit UI logic. All business logic resides inside the `src/` package, keeping the user interface separate from the application logic.

### Benefits of this Architecture

- Separation of concerns
- Modular and maintainable codebase
- Easily testable components
- Simple to add new summarization strategies
- Clear separation between UI and business logic


## Tech Stack

| Layer | Technology |
|--------|------------|
| **LLM Provider** | Groq (`openai/gpt-oss-120b`) |
| **LLM Framework** | LangChain (`langchain-core`, `langchain-groq`, `langchain-text-splitters`) |
| **Embeddings (Semantic Deduplication)** | HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Frontend/UI** | Streamlit |
| **YouTube Content Extraction** | `youtube-transcript-api` |
| **Website Content Extraction** | `trafilatura`, `requests`, `beautifulsoup4` (fallback) |
| **Testing Framework** | `pytest`, `pytest-mock` |

### Why these technologies?

- **Groq** provides fast LLM inference for generating summaries.
- **LangChain** handles document processing, prompt templates, text splitting, and LLM orchestration.
- **HuggingFace Sentence Transformers** generate embeddings used to detect and remove semantically similar chunks before summarization.
- **Streamlit** provides a lightweight web interface for interacting with the summarization pipeline.
- **youtube-transcript-api** extracts transcripts directly from YouTube videos.
- **Trafilatura** extracts the main textual content from websites, while **BeautifulSoup** acts as a fallback when required.
- **Pytest** is used for automated unit testing across the project.

### Note on LangChain

This project uses the modular LangChain packages:

- `langchain-core`
- `langchain-groq`
- `langchain-text-splitters`

instead of the monolithic `langchain` package.

In LangChain 1.x, the legacy `load_summarize_chain` helper and its Stuff, Map-Reduce, and Refine chain implementations were moved to the separate `langchain-classic` package. Instead of relying on those legacy abstractions, this project implements all three summarization strategies directly using LangChain's LCEL (`prompt | llm | parser`) pipeline.

This approach makes each strategy easier to understand, customize, test, and extend while staying aligned with LangChain's current architecture.

## Model Selection

### Primary LLM

**Model:** `openai/gpt-oss-120b` (via Groq)

The application uses **`openai/gpt-oss-120b`** as its default summarization model because it provides a strong balance between context length, reasoning capability, and inference speed.

### Why this model?

- **131K token context window**, allowing the Stuff strategy to summarize most content in a single request while giving Map-Reduce and Refine ample context for each chunk.
- **Strong reasoning and structured-output capabilities**, helping the model consistently generate well-organized summaries.
- **Recommended general-purpose model on Groq**, replacing older models such as `llama-3.3-70b-versatile`, which have been deprecated.
- **Fast inference on Groq's LPU hardware**, providing low-latency responses despite the model's large size.

### Alternative Models Considered

| Model | Reason Not Selected |
|--------|---------------------|
| `openai/gpt-oss-20b` | Smaller, faster, and cheaper. A good candidate for a future **Fast Mode**, but the 120B model consistently provides higher-quality summaries. |
| `qwen/qwen3-32b` and newer Qwen models | Strong benchmark performance, but higher cost and additional complexity than required for this project. |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Excellent inference speed and multimodal support, but multimodal capabilities are not needed because this application processes only text. |

### Selection Rationale

The model was selected based on its published specifications, Groq's current recommendations, and suitability for long-context summarization workloads.

**Note:** No formal benchmark was performed on this project's own dataset. The selection is based on practical engineering considerations rather than a measured comparison across multiple models.

## Summarization Strategies

The application supports three interchangeable summarization strategies. Each strategy is optimized for different content lengths and use cases.

| Strategy | How It Works | Best For | Trade-offs |
|----------|--------------|----------|------------|
| **Stuff** | Combines all chunks into a single prompt and generates one summary using a single Groq API call. | Short to medium-length content | Fastest and simplest approach, but limited by the model's maximum context window. |
| **Map-Reduce** | Summarizes each chunk independently (Map) using concurrent LLM calls, then combines those intermediate summaries into one final summary (Reduce). | Long documents and articles with relatively independent sections | Scales well for large inputs and reduces processing time, but each chunk is summarized without surrounding context. |
| **Refine** | Generates an initial summary from the first chunk, then sequentially updates that summary as each remaining chunk is processed. | Long-form content with strong narrative flow or chronological order | Preserves context better than Map-Reduce but requires sequential LLM calls, making it slower for large inputs. |

### Strategy Selection Logic

- **Stuff** proactively checks the estimated input size before making an LLM request. If the content is too large for a single prompt, it raises a clear error recommending **Map-Reduce** or **Refine** instead of allowing a cryptic context-length failure.
- **Map-Reduce** performs the same validation before its final Reduce step. In rare cases where the combined intermediate summaries still exceed the model's context window, the strategy reports a user-friendly error rather than failing unexpectedly.
- **Refine** processes chunks sequentially to preserve context across the document, but it does not currently support checkpointing or recovery if an intermediate LLM call fails.

A hierarchical (recursive) Reduce implementation for extremely large documents is **not included** in the current version and is listed under **Future Improvements**.

## YouTube Pipeline

The YouTube pipeline extracts transcripts from supported YouTube videos and converts them into a common document format for downstream processing.

1. Validate the input URL and extract the video ID.
   - Supported URL formats include:
     - `watch?v=`
     - `youtu.be/`
     - `/embed/`
     - `/shorts/`
     - `/live/`

2. Fetch the transcript using the current instance-based API provided by `youtube-transcript-api`:
   ```python
   YouTubeTranscriptApi().fetch(...)
   ```

3. Convert the transcript into a LangChain `Document` with metadata for the processing pipeline.

4. Handle all supported failure scenarios with clear, user-friendly error messages, including:
   - Invalid or unavailable videos
   - Disabled transcripts
   - Missing transcripts
   - Age-restricted videos
   - Blocked or rate-limited transcript requests

5. Continue the standard processing pipeline:
   - Text cleaning
   - Chunking
   - Semantic deduplication
   - Summarization

### Current Limitation

Only English-language transcripts are currently supported.

During deployment, YouTube transcript requests from **Streamlit Cloud** may be blocked because they originate from shared cloud IP addresses. This is an external limitation of YouTube's transcript service rather than the application itself. The same videos work correctly when the application is run locally.

## Website Pipeline

The website pipeline extracts the primary textual content from web pages and prepares it for summarization.

1. Validate the input URL.

2. Fetch the webpage using `requests` with:
   - A realistic User-Agent header
   - Configurable request timeout

3. Extract the main article content using `trafilatura`, which removes common boilerplate such as:
   - Navigation menus
   - Headers and footers
   - Advertisements
   - Other non-content elements

4. If `trafilatura` extracts too little content, fall back to a `BeautifulSoup`-based parser that:
   - Removes known non-content HTML elements
   - Extracts text from `<p>` tags

5. Convert the extracted content into a LangChain `Document` for downstream processing.

6. Handle all supported failure scenarios with clear, user-friendly error messages, including:
   - Invalid URLs
   - Connection failures
   - Request timeouts
   - HTTP errors
   - Non-HTML responses
   - Pages with insufficient extractable content

### Current Limitation

The application does not use a headless browser. As a result, JavaScript-rendered websites, heavily protected pages, and some paywalled content may not expose enough server-side HTML for successful extraction.

## Chunking Strategy

The application uses LangChain's `RecursiveCharacterTextSplitter` to divide large documents into manageable chunks before summarization.

Unlike a simple fixed-length splitter, it attempts to preserve natural text boundaries by splitting in the following order:

- Paragraphs
- Sentences
- Words
- Characters (only as a last resort)

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CHUNK_SIZE` | `4000` characters | Keeps each chunk large enough to preserve context while remaining suitable for LLM processing. |
| `CHUNK_OVERLAP` | `400` characters (10%) | Preserves context across chunk boundaries and reduces information loss between adjacent chunks. |

A chunk size of **4000 characters** corresponds to roughly **1,000 tokens** (using an approximate conversion of 4 characters per token). This provides a practical balance between context preservation, processing speed, and API cost.

### Design Rationale

The chosen chunk size and overlap values are intended to:

- Preserve semantic context across chunks
- Reduce abrupt context loss at chunk boundaries
- Improve summary quality for long documents
- Keep individual LLM requests efficient for the Map-Reduce and Refine strategies

These values are engineering defaults based on practical reasoning rather than empirical tuning against a labeled evaluation dataset.

## Prompt Engineering

All prompt templates are centralized in `src/summarization/prompts.py`. No other part of the codebase constructs prompts directly, making prompt management consistent and easier to maintain.

### Design Principles

- All summarization strategies share a common structured output format.
- A single factuality and constraint block is reused across strategies to encourage consistent, grounded summaries.
- Prompt definitions are centralized, preventing prompt drift as the project evolves.
- Each strategy only customizes the behavior required for its summarization workflow.

### Strategy-Specific Prompts

- **Stuff** uses a single prompt that summarizes the entire document in one LLM call.
- **Map-Reduce**
  - The **Map** step generates concise plain-text summaries for individual chunks.
  - The **Reduce** step combines those intermediate summaries into the final structured output.
- **Refine** starts with an initial summary and repeatedly updates it as each new chunk is processed.

The Map step intentionally produces **plain text instead of the final structured format** because its output is an intermediate artifact that is consumed only by the Reduce step.

For a detailed explanation of each prompt and the reasoning behind its design, see `docs/ARCHITECTURE.md`.

## Role of Each Core Technology

### LangChain

LangChain provides the core orchestration layer for the application, including:

- `Document` objects for representing extracted content
- `RecursiveCharacterTextSplitter` for chunking large documents
- `ChatPromptTemplate` for prompt management
- `ChatGroq` for connecting to Groq-hosted LLMs
- LCEL (`prompt | llm | parser`) pipelines for implementing the Stuff, Map-Reduce, and Refine summarization strategies

### Groq

Groq serves as the LLM inference provider for the application.

It executes every summarization request through the `langchain-groq` integration, providing fast inference for all three summarization strategies.

### HuggingFace

The project uses the local Sentence Transformer model:

`sentence-transformers/all-MiniLM-L6-v2`

Its role is **only** semantic deduplication.

Before sending chunks to the LLM, embeddings are generated to identify and remove near-duplicate content, reducing redundant summarization requests—particularly for YouTube transcripts that often contain repeated intros, sponsor messages, or recurring phrases.

This model does **not** generate summaries. It was chosen because it is lightweight (approximately **90 MB**), runs locally without a GPU or HuggingFace API token, and works well within the resource limits of Streamlit Cloud.

### Streamlit

Streamlit is responsible only for the user interface.

It handles:

- URL input
- Strategy selection
- Pipeline progress display
- Error presentation
- Summary rendering

All extraction, processing, and summarization logic remains inside the `src/` package, keeping the presentation layer separate from the application's business logic.

## Error Handling

The application uses custom, domain-specific exceptions to ensure users receive clear, actionable error messages while preserving detailed technical information for debugging.

### Design

- Each project-specific exception exposes a `.user_message` that is safe to display in the UI.
- Technical exception details are written to the application logs instead of being shown to the user.
- This separation keeps the interface user-friendly while providing sufficient diagnostic information during development.

### Handled Failure Scenarios

The application provides dedicated error handling for:

- Invalid or unsupported URLs
- Missing or disabled YouTube transcripts
- Age-restricted or unavailable YouTube videos
- YouTube transcript blocking or rate limiting
- Website connection failures
- Request timeouts
- HTTP errors
- Non-HTML responses
- Pages with insufficient extractable content
- Content exceeding a strategy's supported context size
- Groq API errors, including:
  - Authentication failures
  - Rate-limit errors
  - Request timeouts
  - Connection failures
  - Bad requests
  - Other HTTP status errors

This approach allows the application to fail gracefully, present meaningful feedback to users, and avoid exposing internal implementation details or sensitive information.

## Security

The application follows basic security best practices for managing credentials and sensitive information.

- API keys are **never hardcoded** into the source code.
- During local development, secrets are loaded from a `.env` file (see `.env.example`).
- In Streamlit Cloud deployments, secrets are managed through **Streamlit Secrets** instead of source-controlled files.
- The project's `.gitignore` prevents sensitive files and unnecessary artifacts from being committed, including:
  - `.env`
  - `.streamlit/secrets.toml`
  - Python virtual environments
  - Cache directories
- Application logs never include API keys or other sensitive credentials. By design, `config.py` is the only module responsible for loading secrets, and no logging statements expose secret values.

These practices help keep sensitive credentials out of version control while maintaining a clean separation between configuration and application code.

## Testing

The project includes **135 automated tests** covering the complete application pipeline.

### Test Coverage

- URL validation and YouTube video ID extraction (all supported URL formats)
- YouTube and website extraction (all documented failure scenarios using mocks)
- Text cleaning, chunking, and metadata preservation
- Semantic deduplication with deterministic test embeddings
- Groq/LangChain error handling (`safe_invoke` and `safe_batch`)
- All three summarization strategies:
  - Stuff
  - Map-Reduce
  - Refine
- Strategy Factory and strategy selection logic

All external dependencies (Groq API, network requests, YouTube APIs, and HuggingFace models) are mocked during testing, ensuring the test suite runs without API keys or internet access.

Run the complete test suite:

```bash
pytest -v
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/anaskazi-dev-mind/genai-youtube-website-summarizer.git
cd genai-youtube-website-summarizer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create the environment file

**Linux / macOS**

```bash
cp .env.example .env
```

**Windows**

```cmd
copy .env.example .env
```

Open the `.env` file and add your Groq API key.

### 6. Run the application

```bash
streamlit run app.py
```

---

## Environment Variables

The project uses environment variables for configuration.

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Groq API key from https://console.groq.com |
| `GROQ_MODEL_NAME` | No | Override the default Groq model |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

See `.env.example` for the complete configuration.

---

## GitHub & Deployment Workflow

The development workflow follows this pipeline:

```
Local Development
        │
        ▼
Git Commit
        │
        ▼
GitHub Repository
        │
        ▼
Streamlit Cloud Deployment
        │
        ▼
Live Application
```

GitHub stores the source code and version history, while Streamlit Cloud hosts the deployed application.

---

## Streamlit Cloud Deployment

1. Push the project to GitHub.
2. Open https://share.streamlit.io.
3. Create a new Streamlit application.
4. Select:
   - Repository: `anaskazi-dev-mind/genai-youtube-website-summarizer`
   - Branch: `main`
   - Main file: `app.py`
5. Go to **Settings → Secrets** and add:

```toml
GROQ_API_KEY = "your-real-api-key"
```

6. Deploy the application.

> **Note:** Website summarization works successfully on Streamlit Cloud. YouTube transcript extraction may fail because YouTube blocks transcript requests originating from some shared cloud IP addresses. The same functionality works correctly in the local environment.

## Limitations

- YouTube transcript requests may be blocked or rate-limited from shared cloud IP addresses, independent of this application's implementation.
- During deployment testing, YouTube summarization worked correctly in the local environment but may fail on Streamlit Cloud because YouTube blocks transcript requests from many shared cloud IPs. Website summarization is not affected.
- Website extraction is best-effort. JavaScript-heavy websites, paywalled content, and anti-bot protections may prevent successful content extraction.
- Only English-language YouTube transcripts are currently supported.
- The Stuff strategy and the Reduce phase of Map-Reduce rely on a single LLM call. Extremely large inputs may still exceed the model's context window because hierarchical/recursive reduction is not yet implemented.
- The Refine strategy does not support checkpointing, so a failure during processing requires restarting the summarization.
- Chunk size, overlap, and deduplication threshold are based on practical defaults and have not been tuned using a labeled evaluation dataset.
- No formal benchmark has been conducted to compare summary quality across different LLMs.

## Future Improvements

- Support multilingual YouTube transcripts with automatic translation.
- Implement hierarchical/recursive reduction for extremely large documents.
- Add checkpointing and retry support for the Refine strategy.
- Use LangChain's `.with_structured_output()` for schema-based responses instead of prompt-based Markdown formatting.
- Add a headless browser fallback for JavaScript-rendered websites.
- Introduce a faster summarization mode using `openai/gpt-oss-20b`.

## Screenshots

### Home Page

![AI YouTube & Website Summarizer](Screenshot%202026-08-14%20155701.png)

### Generated Summary

![AI YouTube & Website Summarizer](<Screenshot 2026-08-14 155720.png>)

## Live Demo

**Live Application:**  
https://genai-youtube-website-summarizer-ezjvepfm9fc7dzy9dbtbdo.streamlit.app/

> **Note:** Website summarization works successfully in the deployed application. YouTube transcript extraction may fail on Streamlit Cloud because YouTube blocks transcript requests from many shared cloud IP addresses. The same functionality works correctly when the application is run locally.
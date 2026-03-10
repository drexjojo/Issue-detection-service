# Intelligent Issue Detection Service

An AI-powered service that analyzes user requirements to detect software issues, extract structured information, generate professional descriptions, and find duplicates across multiple ticketing systems.

## Architecture

```
POST /analyze  →  FastAPI  →  AnalysisPipeline
                                  │
                             1. Classify (Ollama LLM)
                                  │
                             2. Extract (Ollama function calling)
                                  │
                             3. Generate (Ollama LLM)
                                  │
                             4. Duplicate Detection (Retrieve + Rerank)
                                ├── Retrieve: ChromaDB bi-encoder search
                                │     └── all-MiniLM-L6-v2 (384-dim, cosine)
                                └── Rerank: Cross-encoder pairwise scoring
                                      └── ms-marco-MiniLM-L-6-v2
                                  │
                             5. Threshold + Top-K → Response
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| **MCP Server** | `src/mcp_server/server.py` | Wraps local ticketing REST API as MCP tools |
| **FastAPI Service** | `src/api/` | `POST /analyze` endpoint |
| **LLM Client** | `src/core/llm_client.py` | Thin async Ollama wrapper using httpx |
| **Classifier** | `src/core/classifier.py` | LLM-based issue vs. non-issue classification |
| **Extractor** | `src/core/extractor.py` | Structured field extraction via Ollama function calling |
| **Generator** | `src/core/generator.py` | Professional issue description generation |
| **Duplicate Detector** | `src/core/duplicate_detector.py` | Two-stage retrieve + rerank duplicate search |
| **Embedding Store** | `src/embeddings/store.py` | ChromaDB-backed vector index with sentence-transformers |
| **MCP Clients** | `src/clients/` | Stdio (local Klaire) and HTTP (Linear) MCP clients |

## Design Decisions

### 1. Two-Stage Duplicate Detection (Retrieve + Rerank)

Analysis of the 155 issues across both ticketing systems revealed that **333 cross-title duplicate pairs** exist — issues with completely different titles but identical or near-identical descriptions. A keyword-based title search misses all of these.

This motivated a two-stage approach:

- **Stage 1 — Retrieve (bi-encoder)**: `all-MiniLM-L6-v2` encodes the query into a 384-dimensional vector. ChromaDB performs approximate nearest neighbor search (HNSW) across all pre-indexed issues from both sources, returning the top-20 candidates. This is fast (sub-millisecond for 155 vectors) and high-recall.

- **Stage 2 — Rerank (cross-encoder)**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores each (query, candidate) pair with full cross-attention. Unlike the bi-encoder which encodes texts independently, the cross-encoder processes both texts jointly, producing more precise relevance scores. Raw logits are normalized to [0, 1] via sigmoid, then filtered by threshold.

The bi-encoder is fast but approximate; the cross-encoder is precise but O(n) per query — combining them gives both speed and accuracy.

### 2. ChromaDB as Unified Vector Store

All issues from both local Klaire and Linear are stored in a single ChromaDB collection with source metadata. The database schema:

| Field | Example | Purpose |
|-------|---------|---------|
| **id** | `"local:6bb14c39-..."` | Source-prefixed to avoid collisions |
| **embedding** | `[0.023, -0.118, ...]` (384 floats) | Pre-computed by `all-MiniLM-L6-v2` |
| **document** | `"KYC verification pending... Government-issued ID..."` | `"{title} {description}"` for reranking |
| **metadata** | `{"source": "local", "title": "...", "issue_id": "..."}` | Filtering and display |

ChromaDB persists to `data/chroma_db/`. On subsequent startups, if data already exists for a source, indexing is skipped entirely.

### 3. Ollama Function Calling for Extraction

Instead of asking the LLM to return JSON text (which can fail to parse), we use Ollama's native tool/function calling support (OpenAI-compatible format). This produces structured tool call responses that map directly to our Pydantic schema — no parsing ambiguity.

### 4. MCP Client Inside FastAPI

The FastAPI service itself acts as an MCP client. This:
- Gives us deterministic control over the search workflow

### 5. Pre-computed Embeddings at Startup

All issues from both sources are fetched via MCP and embedded during FastAPI's lifespan startup. This means:
- No per-request embedding computation latency
- ChromaDB handles persistence — data survives restarts without re-fetching
- 155 × 384-dim vectors = ~240KB in the HNSW index — negligible

### 6. Linear Integration

Linear issues are indexed alongside local issues in the same ChromaDB collection. At query time, duplicate search returns results from both sources with `source` metadata indicating the origin. A `LINEAR_API_KEY` is required in the `.env` file.

### 7. Local LLM (Zero External API Dependencies)

By using Ollama with `qwen2.5:7b`, the entire pipeline runs locally with no external API keys or rate limits. The model provides excellent function calling support and runs efficiently.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) with a function-calling capable model

### Installation

```bash
cd issue-detection-service

# Install Ollama and pull the model
brew install ollama
ollama serve  # starts on localhost:11434
ollama pull qwen2.5:7b

# Install Python dependencies
uv sync

# Install dev dependencies (for tests)
uv sync --extra dev

# Configure environment
cp .env.example .env
# Edit .env to add LINEAR_API_KEY (required)
```

### Running the Service

```bash
# Start the FastAPI server
uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The service will:
1. Open ChromaDB vector store at `data/chroma_db/`
2. Connect to local Klaire MCP server (stdio) and Linear MCP server (HTTP)
3. Index issues from each source not yet in ChromaDB (skipped on subsequent runs)
4. Load the cross-encoder reranker model
5. Be ready to accept requests

### Running the MCP Server Standalone

```bash
uv run python src/mcp_server/server.py
```

This runs the MCP server over stdio — useful for testing with the [MCP Inspector](https://github.com/modelcontextprotocol/inspector).

## API Usage

### `POST /analyze`

```bash
# Issue example
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"requirement": "The login page returns a 500 Internal Server Error when users try to submit their credentials. This has been happening since the last deployment."}'
```

**Response (issue detected):**
```json
{
  "is_issue": true,
  "classification_confidence": 0.95,
  "extracted_fields": {
    "title": "Login page returns 500 error on credential submission",
    "category": "bug",
    "priority": "HIGH",
    "component": "Authentication Service",
    "actual_behavior": "Server returns 500 Internal Server Error"
  },
  "generated_description": "## Login page returns 500 error...",
  "duplicates": [
    {
      "issue_id": "abc-123",
      "title": "Login authentication failing",
      "similarity_score": 0.87,
      "source": "local",
      "match_method": "semantic"
    }
  ]
}
```

```bash
# Non-issue example
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"requirement": "Can we schedule a team meeting for Friday?"}'
```

**Response (not an issue):**
```json
{
  "is_issue": false,
  "classification_confidence": 0.92,
  "reason": "This is a meeting scheduling request, not a software issue",
  "duplicates": []
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/test_mcp_server.py -v
uv run pytest tests/test_classifier.py -v
uv run pytest tests/test_pipeline.py -v
uv run pytest tests/test_api.py -v
```

## Configuration

All settings are configured via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LINEAR_API_KEY` | *(required)* | Linear API key for cross-system duplicate detection |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | LLM model for classification, extraction, generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Bi-encoder model for semantic retrieval |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking |
| `SIMILARITY_THRESHOLD` | `0.5` | Minimum sigmoid-normalized score to report a duplicate |
| `MAX_DUPLICATES` | `5` | Maximum duplicate matches to return |

## Assumptions

1. **Local ticketing API is read-only** — The MCP server exposes read-only tools (list, search, get). The API does not support writes.
2. **qwen2.5:7b** is used for LLM tasks — excellent function calling support, runs locally on Apple Silicon with no API keys needed. Model can be changed via `OLLAMA_MODEL` env var.
3. **Similarity threshold of 0.5** — This is the sigmoid midpoint for cross-encoder logits; scores above 0.5 indicate the reranker considers the candidate relevant. Adjustable via `SIMILARITY_THRESHOLD` env var.

## Tech Stack

- **FastAPI** + **uvicorn** — async web framework
- **MCP Python SDK** (`mcp`) — Model Context Protocol server + client
- **Ollama** (`qwen2.5:7b`) — local LLM for classification, extraction, generation
- **ChromaDB** — persistent vector database with HNSW indexing
- **sentence-transformers** — `all-MiniLM-L6-v2` bi-encoder + `ms-marco-MiniLM-L-6-v2` cross-encoder
- **httpx** — async HTTP client (also used as Ollama API client)
- **Pydantic** — data validation and settings management
- **pytest** — testing framework

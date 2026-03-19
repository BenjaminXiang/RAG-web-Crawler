# RAG Web Crawler

A complete pipeline for crawling web pages, converting to structured Markdown, storing as vector embeddings in Milvus, and querying via CLI or REST API.

## Features

- **Web Crawling** — async crawling via crawl4ai with rate limiting, robots.txt compliance, and configurable concurrency
- **Document Processing** — HTML to Markdown conversion (LLM-enhanced or rule-based), semantic chunking, link/attachment extraction
- **Vector Storage** — Milvus-backed storage with pluggable embeddings (sentence-transformers or OpenAI)
- **Hybrid Search** — vector similarity + BM25 keyword search with fusion ranking and scalar filters
- **REST API** — FastAPI endpoints for query, crawl, and health check
- **LLM Augmented Answers** — optional retrieval-augmented generation with source citations

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Start Milvus

```bash
docker compose up -d
```

### 3. Crawl

```bash
# Single URL
rag-crawler crawl --url https://example.com

# Multiple URLs from a file (one URL per line)
rag-crawler crawl --urls urls.txt

# Crawl without storing to Milvus (Markdown output only)
rag-crawler crawl --url https://example.com --no-store
```

### 4. Query

```bash
# Semantic search (hybrid mode by default)
rag-crawler query "what is RAG"

# Vector-only search, top 5 results
rag-crawler query "machine learning" --mode vector --top-k 5

# Keyword search with URL filter
rag-crawler query "installation guide" --mode keyword --url-filter https://docs.example.com

# LLM augmented answer with citations
rag-crawler query "how to deploy" --llm
```

### 5. REST API

```bash
uvicorn rag_crawler.api.app:app --host 0.0.0.0 --port 8000
```

## CLI Reference

### `rag-crawler crawl`

Crawl URLs, convert to Markdown, and store in vector DB.

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | — | URL to crawl (repeatable) |
| `--urls` | — | File with URLs, one per line |
| `-o, --output` | `./output` | Markdown output directory |
| `--store / --no-store` | `--store` | Store documents in Milvus |
| `--config` | `config.yaml` | Config file path |

### `rag-crawler query`

Query the vector store for relevant text chunks.

| Option | Default | Description |
|--------|---------|-------------|
| `QUERY_TEXT` | — | Search query (positional argument) |
| `-k, --top-k` | `10` | Number of results |
| `-m, --mode` | `hybrid` | Search mode: `hybrid`, `vector`, `keyword` |
| `--url-filter` | — | Filter by source URL (repeatable) |
| `--after` | — | Filter: crawled after ISO timestamp |
| `--before` | — | Filter: crawled before ISO timestamp |
| `--llm / --no-llm` | `--no-llm` | Enable LLM augmented answer |
| `--config` | `config.yaml` | Config file path |

## REST API Endpoints

### `GET /api/health`

Health check. Returns `{"status": "ok", "version": "0.1.0"}`.

### `POST /api/query`

```json
{
  "query": "search text",
  "top_k": 10,
  "mode": "hybrid",
  "filter_urls": ["https://example.com"],
  "crawled_after": "2026-01-01",
  "crawled_before": "2026-12-31",
  "llm": false
}
```

Returns `{"results": [...], "answer": null}`.

### `POST /api/crawl`

```json
{
  "urls": ["https://example.com", "https://example.org"]
}
```

Returns `{"task_id": "uuid", "status": "pending"}`. Crawl runs asynchronously.

### `GET /api/crawl/{task_id}`

Returns task status: `pending`, `running`, `completed`, or `failed`.

## Configuration

All settings are in `config.yaml`:

```yaml
crawler:
  rate_limit: 1.0          # requests per second
  concurrency: 1           # max concurrent requests
  timeout: 30              # seconds
  user_agent: "RAG-web-Crawler/0.1.0"
  respect_robots_txt: true

processor:
  chunk_size: 512          # max tokens per chunk
  chunk_overlap: 50        # overlap tokens between chunks
  output_dir: "./output/markdown"

store:
  milvus:
    uri: "http://localhost:19530"
    collection_name: "rag_chunks"
  embedding:
    provider: "local"      # "local" or "openai"
    model: "all-MiniLM-L6-v2"
    # openai_model: "text-embedding-3-small"
  export_dir: "./output/jsonl"

api:
  host: "0.0.0.0"
  port: 8000

llm:
  provider: "local"        # LLM provider for augmented answers
  base_url: "http://localhost:8080/v1"
  api_key: "your-api-key"
  model: "your-model"
```

### Embedding Providers

- **local** (default): Uses `sentence-transformers` with `all-MiniLM-L6-v2`. No API key required.
- **openai**: Set `provider: "openai"` and `openai_model: "text-embedding-3-small"`. Requires `OPENAI_API_KEY` env var.

## Architecture

```
URL list ──> Crawler ──> Document Processor ──> Vector Store (Milvus)
               │               │                      │
               │          Markdown files          Embeddings
               │          + YAML front matter     + metadata
               │                                      │
               └──────────────────────────────────> Query Interface
                                                   (CLI / REST API)
```

| Module | Path | Description |
|--------|------|-------------|
| Crawler | `src/rag_crawler/crawler/` | Async web crawling, URL parsing, robots.txt |
| Processor | `src/rag_crawler/processor/` | HTML→Markdown, chunking, link extraction |
| Store | `src/rag_crawler/store/` | Milvus client, embeddings, search, JSONL export |
| API | `src/rag_crawler/api/` | FastAPI endpoints, LLM answer generation |
| CLI | `src/rag_crawler/cli.py` | Click-based command line interface |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

MIT

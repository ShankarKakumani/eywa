<p align="center">
  <img src="assets/logo.png" alt="Eywa Logo" width="200">
</p>

# Eywa

[![Release](https://github.com/ShankarKakumani/eywa/actions/workflows/release.yml/badge.svg)](https://github.com/ShankarKakumani/eywa/actions/workflows/release.yml)
[![Tests](https://github.com/ShankarKakumani/eywa/actions/workflows/rust.yml/badge.svg)](https://github.com/ShankarKakumani/eywa/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

> The memory your team never loses.

A local-first knowledge base with hybrid search and cross-encoder reranking. Single binary, no external dependencies.

Named after the neural network from Avatar that connects all life and stores collective memory.

<!-- TODO: Add demo GIF here -->
<!-- ![Eywa Demo](docs/demo.gif) -->

## Why Eywa?

| Feature | Description |
|---------|-------------|
| **100% Local** | All processing on your machine. No data leaves, no API keys required. |
| **Hybrid Search** | Vector similarity + BM25 keyword search with convex fusion. |
| **Cross-Encoder Reranking** | Precision filtering using ms-marco-MiniLM. |
| **Single Binary** | Pure Rust. No Python, no Docker, no server processes. |
| **~65ms Latency** | Production-quality search performance. |

## Search Pipeline

```
Query
  │
  ▼
┌─────────────────┐
│  Embed Query    │  bge-base-en-v1.5 (~10ms)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│LanceDB│ │Tantivy│
│Vector │ │ BM25  │
│Top 50 │ │Top 50 │
│(~15ms)│ │(~10ms)│
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│ Convex Fusion   │  α=0.8 vector, β=0.2 BM25
│    Top 20       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Cross-Encoder   │  ms-marco-MiniLM (~30ms)
│ Rerank → Top 5  │
└────────┬────────┘
         ▼
     Results (~65ms total)
```

## Features

- **Hybrid Retrieval** - Combines semantic vector search with BM25 keyword matching
- **Cross-Encoder Reranking** - Pairwise scoring for precision
- **Smart Chunking** - Markdown-aware (headers, sections), paragraph-based for text
- **Contextual Embeddings** - Document/section context prepended before embedding
- **Compressed Storage** - SQLite + zstd for full document retrieval
- **Batched Ingestion** - Accumulates documents before writing to prevent fragmentation
- **MCP Integration** - Works with Claude Desktop and Cursor

## Installation

### Homebrew (macOS/Linux)

```bash
brew tap ShankarKakumani/eywa
brew install eywa
```

### Download Binary

**macOS (Apple Silicon)**
```bash
curl -L https://github.com/ShankarKakumani/eywa/releases/latest/download/eywa-darwin-arm64 -o eywa
chmod +x eywa && sudo mv eywa /usr/local/bin/
```

**macOS (Intel)**
```bash
curl -L https://github.com/ShankarKakumani/eywa/releases/latest/download/eywa-darwin-x64 -o eywa
chmod +x eywa && sudo mv eywa /usr/local/bin/
```

**Linux (x64)**
```bash
curl -L https://github.com/ShankarKakumani/eywa/releases/latest/download/eywa-linux-x64 -o eywa
chmod +x eywa && sudo mv eywa /usr/local/bin/
```

### Build from Source

```bash
git clone https://github.com/ShankarKakumani/eywa.git
cd eywa
cargo build --release
sudo cp target/release/eywa /usr/local/bin/
```

## Quick Start

```bash
# Ingest documents
eywa ingest --source docs /path/to/documents

# Search
eywa search "how does authentication work"

# Start HTTP server
eywa serve --port 3000
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `eywa ingest -s <source> <path>` | Ingest files from path |
| `eywa search <query>` | Search the knowledge base |
| `eywa sources` | List all sources |
| `eywa docs <source>` | List documents in a source |
| `eywa delete <source>` | Delete a source |
| `eywa reset` | Delete all data |
| `eywa serve -p <port>` | Start HTTP server (default: 3000) |
| `eywa mcp` | Start MCP server |
| `eywa info` | Show model and database info |

## HTTP API

### Search
```bash
curl -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication flow", "limit": 5}'
```

### Ingest Documents
```bash
curl -X POST http://localhost:3000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "docs",
    "documents": [
      {"title": "Auth Guide", "content": "..."}
    ]
  }'
```

### Other Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sources` | List all sources |
| GET | `/api/sources/:id/docs` | List documents in source |
| GET | `/api/docs/:id` | Get document by ID |
| DELETE | `/api/docs/:id` | Delete document |
| DELETE | `/api/sources/:id` | Delete source |
| GET | `/api/export` | Export all as zip |
| DELETE | `/api/reset` | Reset all data |

## MCP Integration

Add to your Claude Desktop or Cursor MCP config:

```json
{
  "mcpServers": {
    "eywa": {
      "command": "/usr/local/bin/eywa",
      "args": ["mcp"]
    }
  }
}
```

**Available Tools:**
- `search` - Search the knowledge base
- `list_sources` - List document sources

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              Eywa                                    │
├──────────────────┬──────────────────┬───────────────────────────────┤
│       CLI        │    HTTP API      │         MCP Server            │
│                  │     (Axum)       │       (JSON-RPC stdio)        │
├──────────────────┴──────────────────┴───────────────────────────────┤
│                           Core Library                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  Embedder  │  │  Chunker   │  │  Search    │  │  Reranker  │    │
│  │  (Candle)  │  │ (Markdown, │  │  Engine    │  │ (Candle)   │    │
│  │            │  │  Text, PDF)│  │            │  │            │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                           Storage Layer                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │    LanceDB     │  │    Tantivy     │  │     SQLite     │        │
│  │   (vectors)    │  │    (BM25)      │  │ (content+zstd) │        │
│  └────────────────┘  └────────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Implementation |
|-----------|----------------|
| Language | Rust |
| Embeddings | Candle + bge-base-en-v1.5 (768 dims) |
| Reranking | Candle + ms-marco-MiniLM-L-6-v2 |
| Vector DB | LanceDB (embedded, file-based) |
| BM25 Search | Tantivy |
| Content Store | SQLite + zstd compression |
| HTTP Server | Axum |
| Fusion | Convex combination (α=0.8, β=0.2) |

## Performance

| Stage | Latency |
|-------|---------|
| Embed query | ~10ms |
| Vector search | ~15ms |
| BM25 search | ~10ms |
| Fusion | <1ms |
| Rerank (20 docs) | ~30ms |
| **Total** | **~65ms** |

Tested on Apple M1. Performance varies by hardware.

## Data Storage

```
~/.eywa/
├── data/
│   ├── vectors/      # LanceDB (embeddings)
│   ├── content.db    # SQLite (full documents, zstd compressed)
│   └── tantivy/      # BM25 index
└── models/           # Downloaded embedding models
```

## Supported File Types

| Category | Extensions |
|----------|------------|
| Markdown | `.md` |
| Text | `.txt` |
| Code | `.rs`, `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.go`, `.java`, `.c`, `.cpp`, `.h`, `.hpp`, `.dart`, `.swift`, `.kt`, `.rb`, `.php` |
| Config | `.json`, `.yaml`, `.yml`, `.toml`, `.xml` |
| Web | `.html`, `.css`, `.scss`, `.vue`, `.svelte` |
| Scripts | `.sh`, `.bash`, `.zsh`, `.fish` |
| Database | `.sql` |

## Roadmap

- [x] **Phase 1**: Hybrid search + cross-encoder reranking
- [ ] **Phase 2**: Query understanding, LLM synthesis with citations
- [ ] **Phase 3**: Evaluation metrics, ground truth testing

## Building

```bash
cargo build --release    # Optimized build
cargo test               # Run tests (60 tests)
```

## License

MIT

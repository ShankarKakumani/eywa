# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eywa is a personal knowledge base with local embeddings and vector search. Named after the neural network from Avatar that connects all life and stores collective knowledge. It uses the all-MiniLM-L6-v2 model via Candle (pure Rust) for local embedding generation and LanceDB for file-based vector storage.

## Build and Run Commands

```bash
# Build the project
cargo build

# Build release version (optimized, with LTO)
cargo build --release

# Run CLI
cargo run -- <command>

# Run tests
cargo test

# Run a single test
cargo test <test_name>
```

## CLI Commands

```bash
# Ingest documents from a file or directory
eywa ingest --source <name> <path>

# Search for documents
eywa search "<query>" --limit 5

# List all sources
eywa sources

# Delete a source
eywa delete <source>

# Start HTTP server (default port 3000)
eywa serve --port 3000

# Show model info
eywa info
```

## Architecture

The project builds as both a library (for Flutter integration via flutter_rust_bridge) and a CLI binary.

### Core Components

- **`Eywa`** (`lib.rs`): High-level API wrapping all components with async RwLock-protected state
- **`Embedder`** (`embed.rs`): Candle (pure Rust) wrapper for all-MiniLM-L6-v2 embeddings (384 dimensions), auto-downloads from HuggingFace
- **`VectorDB`** (`db.rs`): LanceDB wrapper for vector storage and cosine similarity search
- **`Ingester`** (`ingest.rs`): Document chunking (1000 chars with 200 overlap) and batch embedding
- **`SearchEngine`** (`search.rs`): Result filtering by score threshold (0.3) and keyword-based reranking

### Data Flow

1. **Ingest**: Files → chunk by lines → generate embeddings in batches of 32 → store in LanceDB with content hash for deduplication
2. **Search**: Query → embed → vector search → filter by score → rerank with keyword boost → return results

### HTTP API (when running `serve`)

- `GET /health` - Health check
- `POST /search` - Search documents (body: `{"query": "...", "limit": 5}`)
- `GET /sources` - List all sources

## Data Storage

- Database: `~/.eywa/data/`
- Model files: Auto-downloaded to HuggingFace cache on first use

## Supported File Types for Ingestion

md, txt, rs, py, js, ts, tsx, jsx, go, java, c, cpp, h, hpp, json, yaml, yml, toml, xml, html, css, scss, sql, sh, bash, zsh, fish, dart, swift, kt, kts, rb, php, vue, svelte

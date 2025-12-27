//! Eywa CLI
//!
//! Commands:
//!   ingest - Ingest documents from a file or directory
//!   search - Search for similar documents
//!   sources - List all sources
//!   serve - Start HTTP server
//!   mcp - Start MCP server (for Claude/Cursor)

use anyhow::Result;
use clap::{Parser, Subcommand};
use dirs;
use eywa::db;
use eywa::{
    create_job_queue, run_download_wizard, run_init, show_status, show_welcome,
    BM25Index, Config, ContentStore, DocumentInput, Embedder, IngestPipeline, InitResult,
    Reranker, SearchEngine, SearchResult, SharedJobQueue, VectorDB,
};
use std::io::Write;
use std::sync::Arc;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "eywa")]
#[command(about = "Personal knowledge base with local embeddings")]
#[command(version)]
struct Cli {
    /// Data directory for storing the database
    #[arg(short, long, default_value = "~/.eywa/data")]
    data_dir: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest documents from a file or directory
    Ingest {
        /// Source ID (name for this collection)
        #[arg(short, long)]
        source: String,

        /// Path to file or directory to ingest
        path: PathBuf,
    },

    /// Search for documents
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Filter by source ID
        #[arg(short, long)]
        source: Option<String>,
    },

    /// List all sources
    Sources,

    /// List documents in a source
    Docs {
        /// Source ID
        source: String,
    },

    /// Delete a source
    Delete {
        /// Source ID to delete
        source: String,
    },

    /// Reset - delete ~/.eywa (config, data, sqlite). Keeps models.
    Reset,

    /// Hard reset - delete everything including downloaded models
    HardReset,

    /// Uninstall - delete all data and show binary removal instructions
    Uninstall,

    /// Start HTTP server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8005")]
        port: u16,
    },

    /// Start MCP server (for Claude/Cursor)
    Mcp,

    /// Show model info
    Info,

    /// Show storage usage (data, models, total)
    Storage,

    /// Run initialization flow (re-configure models)
    Init {
        /// Use default models without prompts (for CI/scripting)
        #[arg(long)]
        default: bool,
    },
}

fn expand_path(path: &str) -> String {
    if path.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return path.replacen("~", &home, 1);
        }
    }
    path.to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let data_dir = expand_path(&cli.data_dir);

    // Ensure data directory exists
    std::fs::create_dir_all(&data_dir)?;

    match cli.command {
        None => {
            // No command = show status or run init if first run
            match Config::load()? {
                Some(config) => {
                    // Show status
                    let db = VectorDB::new(&data_dir).await?;
                    let sources = db.list_sources().await?;
                    let total_chunks: usize = sources.iter().map(|s| s.chunk_count as usize).sum();
                    let total_docs: usize = {
                        let mut count = 0;
                        for source in &sources {
                            count += db.list_documents(&source.name, Some(db::MAX_QUERY_LIMIT)).await?.len();
                        }
                        count
                    };
                    show_status(&config, sources.len(), total_docs, total_chunks);
                }
                None => {
                    // First run - run init
                    show_welcome();
                    match run_init(None)? {
                        InitResult::Configured(config) => {
                            println!("\n\x1b[32m✓\x1b[0m Configuration saved!\n");

                            // Run the TUI download wizard
                            run_download_wizard(&config)?;

                            // Verify models load correctly (uses hf_hub cache)
                            let _embedder = Embedder::new()?;
                            let _reranker = Reranker::new()?;

                            println!("\n\x1b[32m✓\x1b[0m Setup complete! Run 'eywa --help' to get started.");
                        }
                        InitResult::Cancelled => {
                            println!("\nSetup cancelled.");
                        }
                    }
                }
            }
        }

        Some(Commands::Ingest { source, path }) => {
            println!("Initializing embedder...");
            let embedder = Arc::new(Embedder::new()?);

            println!("Connecting to database...");
            let mut db = VectorDB::new(&data_dir).await?;
            let data_path = std::path::Path::new(&data_dir);
            let bm25_index = Arc::new(BM25Index::open(data_path)?);

            println!("Ingesting documents from: {}\n", path.display());
            let pipeline = IngestPipeline::new(embedder, bm25_index);

            let path_str = path.to_string_lossy().to_string();
            let result = pipeline.ingest_from_path(&mut db, data_path, &source, &path_str).await?;

            println!("\nIngestion complete!");
            println!("  Source: {}", result.source_id);
            println!("  Documents created: {}", result.documents_created);
            println!("  Chunks created: {}", result.chunks_created);
            println!("  Chunks skipped (duplicates): {}", result.chunks_skipped);
        }

        Some(Commands::Search { query, limit, source: _ }) => {
            let embedder = Embedder::new()?;
            let db = VectorDB::new(&data_dir).await?;
            let content_store = ContentStore::open(&std::path::Path::new(&data_dir).join("content.db"))?;
            let search_engine = SearchEngine::with_reranker()?;

            println!("Searching for: {}\n", query);

            let query_embedding = embedder.embed(&query)?;
            let chunk_metas = db.search(&query_embedding, 50).await?;

            // Fetch content from SQLite
            let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
            let contents = content_store.get_chunks(&chunk_ids)?;
            let content_map: std::collections::HashMap<String, String> = contents.into_iter().collect();

            // Combine metadata + content
            let results: Vec<SearchResult> = chunk_metas
                .into_iter()
                .filter_map(|meta| {
                    let content = content_map.get(&meta.id)?.clone();
                    Some(SearchResult {
                        id: meta.id,
                        source_id: meta.source_id,
                        title: meta.title,
                        content,
                        file_path: meta.file_path,
                        line_start: meta.line_start,
                        score: meta.score,
                    })
                })
                .collect();

            let results = search_engine.filter_results(results);
            let results = search_engine.rerank(results, &query, limit);

            if results.is_empty() {
                println!("No results found.");
            } else {
                for (i, result) in results.iter().take(limit).enumerate() {
                    println!("{}. [Score: {:.3}]", i + 1, result.score);
                    if let Some(ref title) = result.title {
                        println!("   Title: {}", title);
                    }
                    if let Some(ref file_path) = result.file_path {
                        print!("   File: {}", file_path);
                        if let Some(line) = result.line_start {
                            print!(":{}", line);
                        }
                        println!();
                    }
                    println!("   Source: {}", result.source_id);

                    // Show first 200 chars of content
                    let preview: String = result
                        .content
                        .chars()
                        .take(200)
                        .collect();
                    println!("   Preview: {}...\n", preview.replace('\n', " "));
                }
            }
        }

        Some(Commands::Sources) => {
            let db = VectorDB::new(&data_dir).await?;
            let sources = db.list_sources().await?;

            if sources.is_empty() {
                println!("No sources found. Use 'eywa ingest' to add documents.");
            } else {
                println!("Sources:\n");
                for source in sources {
                    println!("  {} ({} chunks)", source.name, source.chunk_count);
                }
            }
        }

        Some(Commands::Docs { source }) => {
            let db = VectorDB::new(&data_dir).await?;
            let docs = db.list_documents(&source, Some(db::MAX_QUERY_LIMIT)).await?;

            if docs.is_empty() {
                println!("No documents found in source '{}'.", source);
            } else {
                println!("Documents in '{}':\n", source);
                for doc in docs {
                    println!("  {} - {} ({} chunks, {} chars)",
                        doc.id, doc.title, doc.chunk_count, doc.content_length);
                }
            }
        }

        Some(Commands::Delete { source }) => {
            let data_path = std::path::Path::new(&data_dir);
            let db = VectorDB::new(&data_dir).await?;
            let bm25_index = BM25Index::open(data_path)?;
            let content_store = ContentStore::open(&data_path.join("content.db"))?;

            // Get document IDs for SQLite cleanup
            let doc_ids = db.get_document_ids_for_source(&source).await?;
            let doc_id_refs: Vec<&str> = doc_ids.iter().map(|s| s.as_str()).collect();

            // Delete from all stores
            db.delete_source(&source).await?;
            bm25_index.delete_source(&source)?;
            content_store.delete_source(&doc_id_refs)?;

            println!("Deleted source: {}", source);
        }

        Some(Commands::Reset) => {
            let eywa_dir = dirs::home_dir()
                .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?
                .join(".eywa");

            if eywa_dir.exists() {
                std::fs::remove_dir_all(&eywa_dir)?;
                println!("\x1b[32m✓\x1b[0m Deleted ~/.eywa/");
                println!("\nRun 'eywa' to set up again.");
            } else {
                println!("Nothing to reset - ~/.eywa/ does not exist.");
            }
        }

        Some(Commands::HardReset) => {
            // Get paths
            let home = dirs::home_dir()
                .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
            let eywa_dir = home.join(".eywa");
            let hf_cache = home.join(".cache").join("huggingface").join("hub");
            let fastembed_cache = home.join(".fastembed_cache");

            // Show what will be deleted
            println!("\n\x1b[1;31m⚠ HARD RESET\x1b[0m\n");
            println!("This will permanently delete:");
            println!("  • \x1b[33m~/.eywa/\x1b[0m (config, data, content database)");
            println!("  • \x1b[33m~/.cache/huggingface/hub/\x1b[0m (models)");
            println!("  • \x1b[33m~/.fastembed_cache/\x1b[0m (legacy models)");
            println!();

            // Confirmation prompt
            print!("Type '\x1b[1myes\x1b[0m' to confirm: ");
            std::io::stdout().flush()?;

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;

            if input.trim() != "yes" {
                println!("\nAborted. No data was deleted.");
                return Ok(());
            }

            // Delete eywa directory
            if eywa_dir.exists() {
                std::fs::remove_dir_all(&eywa_dir)?;
                println!("\n\x1b[32m✓\x1b[0m Deleted ~/.eywa/");
            } else {
                println!("\n\x1b[90m~/.eywa/ does not exist\x1b[0m");
            }

            // Delete HuggingFace cache
            if hf_cache.exists() {
                std::fs::remove_dir_all(&hf_cache)?;
                println!("\x1b[32m✓\x1b[0m Deleted ~/.cache/huggingface/hub/");
            } else {
                println!("\x1b[90m~/.cache/huggingface/hub/ does not exist\x1b[0m");
            }

            // Delete legacy fastembed cache
            if fastembed_cache.exists() {
                std::fs::remove_dir_all(&fastembed_cache)?;
                println!("\x1b[32m✓\x1b[0m Deleted ~/.fastembed_cache/");
            }

            println!("\n\x1b[32mHard reset complete.\x1b[0m Run 'eywa' to set up again.");
        }

        Some(Commands::Uninstall) => {
            // Get paths
            let home = dirs::home_dir()
                .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
            let eywa_dir = home.join(".eywa");
            let hf_cache = home.join(".cache").join("huggingface").join("hub");
            let fastembed_cache = home.join(".fastembed_cache");

            // Show what will be deleted
            println!("\n\x1b[1;31m⚠ UNINSTALL EYWA\x1b[0m\n");
            println!("This will permanently delete:");
            println!("  • \x1b[33m~/.eywa/\x1b[0m (config, data, content database)");
            println!("  • \x1b[33m~/.cache/huggingface/hub/\x1b[0m (models)");
            println!("  • \x1b[33m~/.fastembed_cache/\x1b[0m (legacy models)");
            println!();

            // Confirmation prompt
            print!("Type '\x1b[1myes\x1b[0m' to confirm: ");
            std::io::stdout().flush()?;

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;

            if input.trim() != "yes" {
                println!("\nAborted. Nothing was deleted.");
                return Ok(());
            }

            // Delete eywa directory
            if eywa_dir.exists() {
                std::fs::remove_dir_all(&eywa_dir)?;
                println!("\n\x1b[32m✓\x1b[0m Deleted ~/.eywa/");
            } else {
                println!("\n\x1b[90m~/.eywa/ does not exist\x1b[0m");
            }

            // Delete HuggingFace cache
            if hf_cache.exists() {
                std::fs::remove_dir_all(&hf_cache)?;
                println!("\x1b[32m✓\x1b[0m Deleted ~/.cache/huggingface/hub/");
            } else {
                println!("\x1b[90m~/.cache/huggingface/hub/ does not exist\x1b[0m");
            }

            // Delete legacy fastembed cache
            if fastembed_cache.exists() {
                std::fs::remove_dir_all(&fastembed_cache)?;
                println!("\x1b[32m✓\x1b[0m Deleted ~/.fastembed_cache/");
            }

            // Show binary removal instructions
            println!("\n\x1b[32mData deleted.\x1b[0m To complete uninstallation, remove the binary:\n");
            println!("  \x1b[36mHomebrew:\x1b[0m  brew uninstall eywa");
            println!("  \x1b[36mCargo:\x1b[0m     cargo uninstall eywa");
            println!("  \x1b[36mManual:\x1b[0m    rm $(which eywa)");
        }

        Some(Commands::Serve { port }) => {
            println!("Starting server on http://localhost:{}...", port);
            run_server(&data_dir, port).await?;
        }

        Some(Commands::Mcp) => {
            run_mcp_server(&data_dir).await?;
        }

        Some(Commands::Info) => {
            println!("Eywa - Personal Knowledge Base\n");

            match Config::load()? {
                Some(config) => {
                    println!("Embedding: {} ({}MB, {} dims)",
                        config.embedding_model.name(),
                        config.embedding_model.size_mb(),
                        config.embedding_model.dimensions()
                    );
                    println!("Reranker:  {} ({}MB)",
                        config.reranker_model.name(),
                        config.reranker_model.size_mb()
                    );
                }
                None => {
                    println!("Not initialized. Run 'eywa' or 'eywa init' to set up.");
                }
            }

            println!("\nDatabase: LanceDB (file-based)");
            println!("Data directory: {}", data_dir);
        }

        Some(Commands::Storage) => {
            // Helper to format bytes nicely
            fn format_bytes(bytes: u64) -> String {
                if bytes >= 1024 * 1024 * 1024 {
                    format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                } else if bytes >= 1024 * 1024 {
                    format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
                } else if bytes >= 1024 {
                    format!("{:.1} KB", bytes as f64 / 1024.0)
                } else {
                    format!("{} B", bytes)
                }
            }

            println!("Eywa Storage Usage\n");

            // Data storage
            let data_path = std::path::Path::new(&data_dir);
            let content_db_bytes = std::fs::metadata(data_path.join("content.db"))
                .map(|m| m.len())
                .unwrap_or(0);
            let vector_db_bytes = lance_db_size(data_path);
            let bm25_index_bytes = dir_size(&data_path.join("tantivy")).unwrap_or(0);
            let data_total = content_db_bytes + vector_db_bytes + bm25_index_bytes;

            println!("\x1b[1mData\x1b[0m");
            println!("  Content DB (SQLite)    {:>12}", format_bytes(content_db_bytes));
            println!("  Vector DB (LanceDB)    {:>12}", format_bytes(vector_db_bytes));
            println!("  BM25 Index (Tantivy)   {:>12}", format_bytes(bm25_index_bytes));
            println!("  \x1b[90m───────────────────────────────\x1b[0m");
            println!("  Subtotal               {:>12}", format_bytes(data_total));

            // Models storage (scan HuggingFace cache)
            let cached_models = scan_hf_cache();
            let models_total: u64 = cached_models.iter().map(|m| m.size_bytes).sum();

            println!("\n\x1b[1mModels\x1b[0m (cached from HuggingFace)");
            if cached_models.is_empty() {
                println!("  No models downloaded yet");
            } else {
                for model in &cached_models {
                    println!("  {:<24} {:>12}", model.name, format_bytes(model.size_bytes));
                }
                println!("  \x1b[90m───────────────────────────────\x1b[0m");
                println!("  Subtotal               {:>12}", format_bytes(models_total));
            }

            // Total
            let grand_total = data_total + models_total;
            println!("\n\x1b[1m═══════════════════════════════════\x1b[0m");
            println!("\x1b[1mTotal                    {:>12}\x1b[0m", format_bytes(grand_total));
        }

        Some(Commands::Init { default }) => {
            // Non-interactive mode for CI/scripting
            if default {
                let config = Config::default();
                config.save()?;
                println!("Configuration saved with defaults.");
                run_download_wizard(&config)?;
                let _embedder = Embedder::new()?;
                let _reranker = Reranker::new()?;
                println!("\nSetup complete!");
                return Ok(());
            }

            let existing = Config::load()?;

            // Check if previous re-indexing was interrupted
            let marker_path = std::path::Path::new(&data_dir).join(".reindex_in_progress");
            let interrupted = marker_path.exists();

            match run_init(existing.as_ref())? {
                InitResult::Configured(config) => {
                    let needs_reindex = existing
                        .map(|e| e.embedding_model != config.embedding_model)
                        .unwrap_or(false);

                    println!("\n\x1b[32m✓\x1b[0m Configuration saved!");

                    if interrupted {
                        println!("\n\x1b[33m!\x1b[0m Previous re-indexing was interrupted. Resuming...\n");
                    }

                    if needs_reindex || interrupted {
                        if needs_reindex && !interrupted {
                            println!("\n\x1b[33m!\x1b[0m Embedding model changed - re-indexing required\n");
                        }

                        // 1. Get document count from SQLite
                        let content_path = std::path::Path::new(&data_dir).join("content.db");
                        let content_store = ContentStore::open(&content_path)?;
                        let doc_count = content_store.document_count()?;

                        if doc_count == 0 {
                            println!("  No documents to re-index.\n");
                            // Just download new models
                            run_download_wizard(&config)?;
                            // Remove marker if it exists
                            std::fs::remove_file(&marker_path).ok();
                        } else {
                            // 2. Get all documents with metadata from SQLite
                            let documents = content_store.get_all_documents_with_metadata()?;
                            println!("  Found {} documents to re-index\n", documents.len());

                            // 3. Download new models
                            run_download_wizard(&config)?;

                            // 4. Initialize new embedder
                            let embedder = Arc::new(Embedder::new()?);
                            let _reranker = Reranker::new()?;

                            // 5. Create marker file before starting (survives interruption)
                            std::fs::write(&marker_path, "")?;

                            // 6. Reset LanceDB and BM25 index (SQLite stays intact with content)
                            let mut db = VectorDB::new(&data_dir).await?;
                            db.reset_all().await?;
                            let data_path = std::path::Path::new(&data_dir);
                            let bm25_index = Arc::new(BM25Index::open(data_path)?);
                            bm25_index.reset()?;

                            // 7. Re-ingest from SQLite
                            println!("\n  Re-indexing documents...\n");
                            let pipeline = IngestPipeline::new(embedder, bm25_index);
                            let mut total_chunks = 0u32;

                            for (i, doc) in documents.iter().enumerate() {
                                // Show progress
                                print!("\r  [{}/{}] {}                              ",
                                    i + 1, documents.len(),
                                    if doc.title.len() > 40 { &doc.title[..40] } else { &doc.title }
                                );
                                std::io::stdout().flush()?;

                                let doc_input = DocumentInput {
                                    content: doc.content.clone(),
                                    title: Some(doc.title.clone()),
                                    file_path: doc.file_path.clone(),
                                };

                                let result = pipeline
                                    .ingest_documents(&mut db, data_path, &doc.source_id, vec![doc_input])
                                    .await?;
                                total_chunks += result.chunks_created;
                            }

                            // 8. Remove marker on successful completion
                            std::fs::remove_file(&marker_path).ok();

                            println!("\n\n\x1b[32m✓\x1b[0m Re-indexed {} documents ({} chunks)\n",
                                documents.len(), total_chunks);
                        }
                    } else {
                        // No re-indexing needed, just download models
                        run_download_wizard(&config)?;
                        let _embedder = Embedder::new()?;
                        let _reranker = Reranker::new()?;
                    }

                    println!("\n\x1b[32m✓\x1b[0m Setup complete!");
                }
                InitResult::Cancelled => {
                    println!("\nInit cancelled. Configuration unchanged.");
                }
            }
        }
    }

    Ok(())
}

/// Create a zip file from documents
fn create_zip(docs: &[eywa::Document]) -> Result<Vec<u8>> {
    use std::io::{Cursor, Write};
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    let mut buffer = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(&mut buffer);
    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    for doc in docs {
        // Create path: source_id/title (sanitize for filesystem)
        let safe_title = doc.title
            .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        let path = format!("{}/{}", doc.source_id, safe_title);

        zip.start_file(&path, options)?;
        zip.write_all(doc.content.as_bytes())?;
    }

    zip.finish()?;
    Ok(buffer.into_inner())
}

/// Extract text content from HTML and convert to Markdown
fn extract_text_from_html(html: &str) -> String {
    html2md::rewrite_html(html, false)
}

/// Extract title from HTML
fn extract_title_from_html(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<title>")?;
    let end = lower[start..].find("</title>")?;
    let title = &html[start + 7..start + end];
    let title = title.trim();
    if title.is_empty() {
        None
    } else {
        Some(title.to_string())
    }
}

/// Calculate total size of a directory recursively
fn dir_size(path: &std::path::Path) -> std::io::Result<u64> {
    let mut total = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                total += dir_size(&path)?;
            } else {
                total += entry.metadata()?.len();
            }
        }
    }
    Ok(total)
}

/// Calculate total size of all LanceDB table directories (.lance) in a path
fn lance_db_size(data_path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(data_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.extension().map_or(false, |ext| ext == "lance") {
                total += dir_size(&path).unwrap_or(0);
            }
        }
    }
    total
}

/// Model info from HuggingFace cache
#[derive(Debug, Clone, serde::Serialize)]
struct CachedModel {
    name: String,
    size_bytes: u64,
}

/// Scan HuggingFace cache directory and return all downloaded models
fn scan_hf_cache() -> Vec<CachedModel> {
    let mut models = Vec::new();

    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return models,
    };

    let hf_cache = home.join(".cache").join("huggingface").join("hub");
    if !hf_cache.exists() {
        return models;
    }

    // HuggingFace cache structure: ~/.cache/huggingface/hub/models--org--name/
    if let Ok(entries) = std::fs::read_dir(&hf_cache) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let dir_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                // Only process model directories (models--org--name)
                if dir_name.starts_with("models--") {
                    // Parse model name from directory: models--org--name -> org/name
                    let name = dir_name
                        .strip_prefix("models--")
                        .unwrap_or(dir_name)
                        .replace("--", "/");

                    let size = dir_size(&path).unwrap_or(0);

                    models.push(CachedModel {
                        name,
                        size_bytes: size,
                    });
                }
            }
        }
    }

    // Sort by size (largest first)
    models.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    models
}

/// Run the HTTP server
async fn run_server(data_dir: &str, port: u16) -> Result<()> {
    use axum::{
        body::Body,
        extract::{DefaultBodyLimit, Path},
        http::{header, StatusCode},
        response::{Json, Response},
        routing::{delete, get, post},
        Router,
    };
    use eywa::{FetchUrlRequest, IngestRequest, SearchRequest};
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower_http::cors::CorsLayer;

    // Shared components
    let embedder = Arc::new(Embedder::new()?);
    let db = Arc::new(RwLock::new(VectorDB::new(data_dir).await?));
    let bm25_index = Arc::new(BM25Index::open(std::path::Path::new(data_dir))?);
    let search_engine = SearchEngine::new();
    let job_db_path = std::path::Path::new(data_dir).join("jobs.db");
    let job_queue = create_job_queue(&job_db_path)?;

    struct AppState {
        embedder: Arc<Embedder>,
        db: Arc<RwLock<VectorDB>>,
        bm25_index: Arc<BM25Index>,
        search_engine: SearchEngine,
        job_queue: SharedJobQueue,
        data_dir: String,
    }

    let state = Arc::new(AppState {
        embedder: Arc::clone(&embedder),
        db: Arc::clone(&db),
        bm25_index: Arc::clone(&bm25_index),
        search_engine,
        job_queue: Arc::clone(&job_queue),
        data_dir: data_dir.to_string(),
    });

    // Spawn background worker for processing queue
    let worker_queue = Arc::clone(&job_queue);
    let worker_embedder = Arc::clone(&embedder);
    let worker_db = Arc::clone(&db);
    let worker_bm25 = Arc::clone(&bm25_index);
    let worker_data_dir = data_dir.to_string();
    tokio::spawn(async move {
        run_queue_worker(worker_queue, worker_embedder, worker_db, worker_bm25, worker_data_dir).await;
    });

    // API Routes
    let api = Router::new()
        // System info (for dashboard)
        .route("/info", get({
            let state = Arc::clone(&state);
            move || {
                let state = Arc::clone(&state);
                async move {
                    // Load config for model info
                    let config = Config::load().ok().flatten();

                    // Get stats from sources
                    let db = state.db.read().await;
                    let sources = db.list_sources().await.unwrap_or_default();

                    let source_count = sources.len();
                    let chunk_count: u64 = sources.iter().map(|s| s.chunk_count).sum();

                    // Get document count from content store
                    let document_count = ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db"))
                        .ok()
                        .and_then(|cs| cs.count_documents().ok())
                        .unwrap_or(0);

                    // Get storage sizes
                    let data_path = std::path::Path::new(&state.data_dir);
                    let content_db_bytes = std::fs::metadata(data_path.join("content.db"))
                        .map(|m| m.len())
                        .unwrap_or(0);
                    let vector_db_bytes = lance_db_size(data_path);
                    let bm25_index_bytes = dir_size(&data_path.join("tantivy")).unwrap_or(0);

                    // Build response
                    let mut response = json!({
                        "stats": {
                            "source_count": source_count,
                            "document_count": document_count,
                            "chunk_count": chunk_count
                        },
                        "storage": {
                            "content_db_bytes": content_db_bytes,
                            "vector_db_bytes": vector_db_bytes,
                            "bm25_index_bytes": bm25_index_bytes
                        }
                    });

                    // Add model info if config exists
                    if let Some(cfg) = config {
                        response["embedding_model"] = json!({
                            "name": cfg.embedding_model.name(),
                            "size_mb": cfg.embedding_model.size_mb(),
                            "dimensions": cfg.embedding_model.dimensions()
                        });
                        response["reranker_model"] = json!({
                            "name": cfg.reranker_model.name(),
                            "size_mb": cfg.reranker_model.size_mb()
                        });
                    }

                    // Add all cached models from HuggingFace
                    let cached_models = scan_hf_cache();
                    let cached_models_json: Vec<_> = cached_models.iter().map(|m| {
                        json!({
                            "name": m.name,
                            "size_bytes": m.size_bytes
                        })
                    }).collect();
                    response["cached_models"] = json!(cached_models_json);

                    (StatusCode::OK, Json(response))
                }
            }
        }))
        // Search
        .route("/search", post({
            let state = Arc::clone(&state);
            move |Json(payload): Json<SearchRequest>| {
                let state = Arc::clone(&state);
                async move {
                    let query_embedding = match state.embedder.embed(&payload.query) {
                        Ok(e) => e,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": e.to_string() })),
                            )
                        }
                    };

                    let db = state.db.read().await;
                    let chunk_metas = match db.search(&query_embedding, payload.limit * 2).await {
                        Ok(r) => r,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": e.to_string() })),
                            )
                        }
                    };

                    // Fetch content from SQLite (open fresh connection)
                    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
                        Ok(cs) => cs,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": e.to_string() })),
                            )
                        }
                    };
                    let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
                    let contents = match content_store.get_chunks(&chunk_ids) {
                        Ok(c) => c,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": e.to_string() })),
                            )
                        }
                    };
                    let content_map: std::collections::HashMap<String, String> = contents.into_iter().collect();

                    // Combine metadata + content
                    let results: Vec<SearchResult> = chunk_metas
                        .into_iter()
                        .filter_map(|meta| {
                            let content = content_map.get(&meta.id)?.clone();
                            Some(SearchResult {
                                id: meta.id,
                                source_id: meta.source_id,
                                title: meta.title,
                                content,
                                file_path: meta.file_path,
                                line_start: meta.line_start,
                                score: meta.score,
                            })
                        })
                        .collect();

                    let results = state.search_engine.filter_results(results);
                    let results = state.search_engine.rerank_with_keywords(results, &payload.query);
                    let results: Vec<_> = results.into_iter().take(payload.limit).collect();
                    let count = results.len();

                    (
                        StatusCode::OK,
                        Json(json!({
                            "query": payload.query,
                            "results": results,
                            "count": count
                        })),
                    )
                }
            }
        }))
        // Ingest documents (uses IngestPipeline for batch processing)
        .route("/ingest", post({
            let state = Arc::clone(&state);
            move |Json(payload): Json<IngestRequest>| {
                let state = Arc::clone(&state);
                async move {
                    let data_dir = std::path::Path::new(&state.data_dir);
                    let mut db = state.db.write().await;
                    let pipeline = IngestPipeline::new(Arc::clone(&state.embedder), Arc::clone(&state.bm25_index));

                    match pipeline.ingest_documents(&mut db, data_dir, &payload.source_id, payload.documents).await {
                        Ok(result) => (StatusCode::OK, Json(json!(result))),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        ),
                    }
                }
            }
        }))
        // Queue documents for background processing (non-blocking) - legacy endpoint
        .route("/queue", post({
            let state = Arc::clone(&state);
            move |Json(payload): Json<IngestRequest>| {
                let state = Arc::clone(&state);
                async move {
                    let result = {
                        let mut queue = state.job_queue.lock().unwrap();
                        queue.queue_documents(&payload.source_id, payload.documents.clone())
                    };
                    match result {
                        Ok(job_id) => {
                            let docs_queued = payload.documents.len() as u32;
                            (StatusCode::ACCEPTED, Json(json!({
                                "job_id": job_id,
                                "docs_queued": docs_queued,
                                "message": format!("Queued {} documents for processing", docs_queued)
                            })))
                        }
                        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))
                    }
                }
            }
        }))
        // Async ingest - same as /queue but clearer naming
        .route("/ingest/async", post({
            let state = Arc::clone(&state);
            move |Json(payload): Json<IngestRequest>| {
                let state = Arc::clone(&state);
                async move {
                    let result = {
                        let mut queue = state.job_queue.lock().unwrap();
                        queue.queue_documents(&payload.source_id, payload.documents.clone())
                    };
                    match result {
                        Ok(job_id) => {
                            let total_docs = payload.documents.len() as u32;
                            (StatusCode::ACCEPTED, Json(json!({
                                "job_id": job_id,
                                "status": "queued",
                                "total_docs": total_docs
                            })))
                        }
                        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))
                    }
                }
            }
        }))
        // List all jobs
        .route("/jobs", get({
            let state = Arc::clone(&state);
            move || {
                let state = Arc::clone(&state);
                async move {
                    let result = {
                        let queue = state.job_queue.lock().unwrap();
                        queue.list_jobs()
                    };
                    match result {
                        Ok(jobs) => (StatusCode::OK, Json(json!({ "jobs": jobs }))),
                        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))
                    }
                }
            }
        }))
        // Get job status
        .route("/jobs/:job_id", get({
            let state = Arc::clone(&state);
            move |Path(job_id): Path<String>| {
                let state = Arc::clone(&state);
                async move {
                    let result = {
                        let queue = state.job_queue.lock().unwrap();
                        queue.get_job(&job_id)
                    };
                    match result {
                        Ok(Some(job)) => (StatusCode::OK, Json(json!(job))),
                        Ok(None) => (StatusCode::NOT_FOUND, Json(json!({ "error": "Job not found" }))),
                        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))
                    }
                }
            }
        }))
        // Get per-document status for a job
        .route("/jobs/:job_id/docs", get({
            let state = Arc::clone(&state);
            move |Path(job_id): Path<String>| {
                let state = Arc::clone(&state);
                async move {
                    let result = {
                        let queue = state.job_queue.lock().unwrap();
                        queue.get_job_docs(&job_id)
                    };
                    match result {
                        Ok(docs) => (StatusCode::OK, Json(json!({ "docs": docs }))),
                        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))
                    }
                }
            }
        }))
        // List sources
        .route("/sources", get({
            let state = Arc::clone(&state);
            move || {
                let state = Arc::clone(&state);
                async move {
                    let db = state.db.read().await;
                    match db.list_sources().await {
                        Ok(sources) => (StatusCode::OK, Json(json!({ "sources": sources }))),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        ),
                    }
                }
            }
        }))
        // Delete source
        .route("/sources/:source_id", delete({
            let state = Arc::clone(&state);
            move |Path(source_id): Path<String>| {
                let state = Arc::clone(&state);
                async move {
                    let db = state.db.read().await;

                    // Delete from LanceDB
                    if let Err(e) = db.delete_source(&source_id).await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        );
                    }

                    // Delete from BM25 index
                    if let Err(e) = state.bm25_index.delete_source(&source_id) {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        );
                    }

                    // Delete from SQLite directly by source_id (open fresh connection)
                    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
                        Ok(cs) => cs,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": e.to_string() })),
                            )
                        }
                    };
                    if let Err(e) = content_store.delete_source_by_source_id(&source_id) {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        );
                    }

                    (StatusCode::OK, Json(json!({ "deleted": source_id })))
                }
            }
        }))
        // List documents in source (from LanceDB - for MCP compatibility)
        // Accepts optional ?limit=N query param (default 10, "all" for unlimited)
        .route("/sources/:source_id/docs", get({
            let state = Arc::clone(&state);
            move |Path(source_id): Path<String>, axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>| {
                let state = Arc::clone(&state);
                async move {
                    // Parse limit from query params
                    let limit = params.get("limit").and_then(|v| {
                        if v == "all" { Some(db::MAX_QUERY_LIMIT) } else { v.parse().ok() }
                    });

                    let db = state.db.read().await;
                    match db.list_documents(&source_id, limit).await {
                        Ok(docs) => (StatusCode::OK, Json(json!({ "documents": docs }))),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        ),
                    }
                }
            }
        }))
        // Get document
        .route("/docs/:doc_id", get({
            let state = Arc::clone(&state);
            move |Path(doc_id): Path<String>| {
                let state = Arc::clone(&state);
                async move {
                    let db = state.db.read().await;
                    let record = match db.get_document(&doc_id).await {
                        Ok(Some(r)) => r,
                        Ok(None) => return (StatusCode::NOT_FOUND, Json(json!({ "error": "Document not found" }))),
                        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
                    };

                    // Fetch content from SQLite (open fresh connection)
                    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
                        Ok(cs) => cs,
                        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
                    };
                    let content = match content_store.get_document(&doc_id) {
                        Ok(Some(c)) => c,
                        Ok(None) => return (StatusCode::NOT_FOUND, Json(json!({ "error": "Document content not found" }))),
                        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
                    };

                    // Combine metadata + content
                    let doc = eywa::Document {
                        id: record.id,
                        source_id: record.source_id,
                        title: record.title,
                        content,
                        file_path: record.file_path,
                        created_at: record.created_at,
                        chunk_count: record.chunk_count,
                    };

                    (StatusCode::OK, Json(json!(doc)))
                }
            }
        }))
        // Delete document
        .route("/docs/:doc_id", delete({
            let state = Arc::clone(&state);
            move |Path(doc_id): Path<String>| {
                let state = Arc::clone(&state);
                async move {
                    // Delete from LanceDB
                    let db = state.db.read().await;
                    if let Err(e) = db.delete_document(&doc_id).await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        );
                    }

                    // Delete from SQLite (open fresh connection)
                    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
                        Ok(cs) => cs,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": e.to_string() })),
                            )
                        }
                    };
                    if let Err(e) = content_store.delete_document(&doc_id) {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        );
                    }

                    (StatusCode::OK, Json(json!({ "deleted": doc_id })))
                }
            }
        }))
        // ─────────────────────────────────────────────────────────────────────
        // SQLite-backed routes for Web UI (not for MCP)
        // ─────────────────────────────────────────────────────────────────────
        // List sources from SQLite (for web UI)
        .route("/sql/sources", get({
            let state = Arc::clone(&state);
            move || {
                let state = Arc::clone(&state);
                async move {
                    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
                        Ok(cs) => cs,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": e.to_string() })),
                            )
                        }
                    };

                    match content_store.list_sources() {
                        Ok(sources) => (StatusCode::OK, Json(json!({ "sources": sources }))),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        ),
                    }
                }
            }
        }))
        // List documents in source from SQLite (for web UI with pagination)
        .route("/sql/sources/:source_id/docs", get({
            let state = Arc::clone(&state);
            move |Path(source_id): Path<String>, axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>| {
                let state = Arc::clone(&state);
                async move {
                    // Parse limit and offset from query params
                    let limit = params.get("limit").and_then(|v| {
                        if v == "all" { None } else { v.parse().ok() }
                    });
                    let offset = params.get("offset").and_then(|v| v.parse().ok());

                    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
                        Ok(cs) => cs,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": e.to_string() })),
                            )
                        }
                    };

                    match content_store.list_documents_by_source(&source_id, limit, offset) {
                        Ok((docs, total)) => (StatusCode::OK, Json(json!({
                            "documents": docs,
                            "total_documents": total
                        }))),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        ),
                    }
                }
            }
        }))
        // Reset all data
        .route("/reset", delete({
            let state = Arc::clone(&state);
            move || {
                let state = Arc::clone(&state);
                async move {
                    // Reset LanceDB
                    let mut db = state.db.write().await;
                    if let Err(e) = db.reset_all().await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        );
                    }

                    // Reset BM25 index
                    if let Err(e) = state.bm25_index.reset() {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        );
                    }

                    // Reset SQLite (open fresh connection)
                    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
                        Ok(cs) => cs,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": e.to_string() })),
                            )
                        }
                    };
                    if let Err(e) = content_store.reset() {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        );
                    }

                    (StatusCode::OK, Json(json!({ "status": "reset complete" })))
                }
            }
        }))
        // Export all documents as zip (from SQLite - source of truth)
        .route("/export", get({
            let state = Arc::clone(&state);
            move || {
                let state = Arc::clone(&state);
                async move {
                    // Get all documents with metadata from SQLite (open fresh connection)
                    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
                        Ok(cs) => cs,
                        Err(e) => {
                            return Response::builder()
                                .status(StatusCode::INTERNAL_SERVER_ERROR)
                                .body(Body::from(format!("Error: {}", e)))
                                .unwrap();
                        }
                    };

                    let doc_rows = match content_store.get_all_documents_with_metadata() {
                        Ok(rows) => rows,
                        Err(e) => {
                            return Response::builder()
                                .status(StatusCode::INTERNAL_SERVER_ERROR)
                                .body(Body::from(format!("Error: {}", e)))
                                .unwrap();
                        }
                    };

                    // Convert to Document structs for zip creation
                    let docs: Vec<eywa::Document> = doc_rows
                        .into_iter()
                        .map(|r| eywa::Document {
                            id: r.id,
                            source_id: r.source_id,
                            title: r.title,
                            content: r.content,
                            file_path: r.file_path,
                            created_at: r.created_at,
                            chunk_count: 0, // Not stored in SQLite
                        })
                        .collect();

                    match create_zip(&docs) {
                        Ok(zip_data) => Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "application/zip")
                            .header(header::CONTENT_DISPOSITION, "attachment; filename=\"eywa-export.zip\"")
                            .body(Body::from(zip_data))
                            .unwrap(),
                        Err(e) => Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .body(Body::from(format!("Error: {}", e)))
                            .unwrap(),
                    }
                }
            }
        }))
        // Export source documents as zip (from SQLite - source of truth)
        .route("/sources/:source_id/export", get({
            let state = Arc::clone(&state);
            move |Path(source_id): Path<String>| {
                let state = Arc::clone(&state);
                async move {
                    // Get all documents with metadata from SQLite (open fresh connection)
                    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
                        Ok(cs) => cs,
                        Err(e) => {
                            return Response::builder()
                                .status(StatusCode::INTERNAL_SERVER_ERROR)
                                .body(Body::from(format!("Error: {}", e)))
                                .unwrap();
                        }
                    };

                    let doc_rows = match content_store.get_all_documents_with_metadata() {
                        Ok(rows) => rows,
                        Err(e) => {
                            return Response::builder()
                                .status(StatusCode::INTERNAL_SERVER_ERROR)
                                .body(Body::from(format!("Error: {}", e)))
                                .unwrap();
                        }
                    };

                    // Filter by source_id and convert to Document structs
                    let docs: Vec<eywa::Document> = doc_rows
                        .into_iter()
                        .filter(|r| r.source_id == source_id)
                        .map(|r| eywa::Document {
                            id: r.id,
                            source_id: r.source_id,
                            title: r.title,
                            content: r.content,
                            file_path: r.file_path,
                            created_at: r.created_at,
                            chunk_count: 0, // Not stored in SQLite
                        })
                        .collect();

                    match create_zip(&docs) {
                        Ok(zip_data) => Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "application/zip")
                            .header(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{}.zip\"", source_id))
                            .body(Body::from(zip_data))
                            .unwrap(),
                        Err(e) => Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .body(Body::from(format!("Error: {}", e)))
                            .unwrap(),
                    }
                }
            }
        }))
        // Fetch URL preview (without ingesting)
        .route("/fetch-preview", post({
            move |Json(payload): Json<serde_json::Value>| {
                async move {
                    let url = match payload.get("url").and_then(|v| v.as_str()) {
                        Some(u) => u.to_string(),
                        None => {
                            return (
                                StatusCode::BAD_REQUEST,
                                Json(json!({ "error": "URL is required" })),
                            )
                        }
                    };

                    // Fetch the URL
                    let client = reqwest::Client::new();
                    let response = match client.get(&url).send().await {
                        Ok(r) => r,
                        Err(e) => {
                            return (
                                StatusCode::BAD_REQUEST,
                                Json(json!({ "error": format!("Failed to fetch URL: {}", e) })),
                            )
                        }
                    };

                    if !response.status().is_success() {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(json!({ "error": format!("URL returned status: {}", response.status()) })),
                        );
                    }

                    let html = match response.text().await {
                        Ok(t) => t,
                        Err(e) => {
                            return (
                                StatusCode::BAD_REQUEST,
                                Json(json!({ "error": format!("Failed to read response: {}", e) })),
                            )
                        }
                    };

                    // Extract text content from HTML
                    let content = extract_text_from_html(&html);
                    let title = extract_title_from_html(&html).unwrap_or_else(|| url.clone());

                    if content.trim().is_empty() {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(json!({ "error": "No text content found in page" })),
                        );
                    }

                    // Return preview without ingesting
                    (StatusCode::OK, Json(json!({
                        "title": title,
                        "content": content,
                        "url": url
                    })))
                }
            }
        }))
        // Fetch URL and ingest content
        .route("/fetch-url", post({
            let state = Arc::clone(&state);
            move |Json(payload): Json<FetchUrlRequest>| {
                let state = Arc::clone(&state);
                async move {
                    // Fetch the URL
                    let client = reqwest::Client::new();
                    let response = match client.get(&payload.url).send().await {
                        Ok(r) => r,
                        Err(e) => {
                            return (
                                StatusCode::BAD_REQUEST,
                                Json(json!({ "error": format!("Failed to fetch URL: {}", e) })),
                            )
                        }
                    };

                    if !response.status().is_success() {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(json!({ "error": format!("URL returned status: {}", response.status()) })),
                        );
                    }

                    let html = match response.text().await {
                        Ok(t) => t,
                        Err(e) => {
                            return (
                                StatusCode::BAD_REQUEST,
                                Json(json!({ "error": format!("Failed to read response: {}", e) })),
                            )
                        }
                    };

                    // Extract text content from HTML (basic approach)
                    let content = extract_text_from_html(&html);
                    let title = extract_title_from_html(&html).unwrap_or_else(|| payload.url.clone());

                    if content.trim().is_empty() {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(json!({ "error": "No text content found in page" })),
                        );
                    }

                    // Ingest the content (uses IngestPipeline for batch processing)
                    let source_id = payload.source_id.unwrap_or_else(|| "web".to_string());
                    let data_dir = std::path::Path::new(&state.data_dir);
                    let mut db = state.db.write().await;
                    let pipeline = IngestPipeline::new(Arc::clone(&state.embedder), Arc::clone(&state.bm25_index));

                    let docs = vec![eywa::DocumentInput {
                        content,
                        title: Some(title.clone()),
                        file_path: Some(payload.url.clone()),
                    }];

                    match pipeline.ingest_documents(&mut db, data_dir, &source_id, docs).await {
                        Ok(result) => (StatusCode::OK, Json(json!({
                            "title": title,
                            "url": payload.url,
                            "documents_created": result.documents_created,
                            "chunks_created": result.chunks_created
                        }))),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": e.to_string() })),
                        ),
                    }
                }
            }
        }));

    let app = Router::new()
        // Web UI v2 (default)
        .route("/", get(|| async {
            axum::response::Html(include_str!("../web/v2/index.html"))
        }))
        .route("/style.css", get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "text/css")],
                include_str!("../web/v2/style.css")
            )
        }))
        .route("/api.js", get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "application/javascript")],
                include_str!("../web/v2/api.js")
            )
        }))
        .route("/app.js", get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "application/javascript")],
                include_str!("../web/v2/app.js")
            )
        }))
        .route("/dashboard.js", get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "application/javascript")],
                include_str!("../web/v2/dashboard.js")
            )
        }))
        .route("/add-docs.js", get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "application/javascript")],
                include_str!("../web/v2/add-docs.js")
            )
        }))
        .route("/explorer.js", get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "application/javascript")],
                include_str!("../web/v2/explorer.js")
            )
        }))
        .route("/jobs.js", get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "application/javascript")],
                include_str!("../web/v2/jobs.js")
            )
        }))
        .route("/favicon.png", get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "image/png")],
                include_bytes!("../web/v2/favicon.png").as_slice()
            )
        }))
        .route("/apple-touch-icon.png", get(|| async {
            (
                [(axum::http::header::CONTENT_TYPE, "image/png")],
                include_bytes!("../web/v2/apple-touch-icon.png").as_slice()
            )
        }))
        // Web UI v1 (legacy)
        .route("/v1", get(|| async {
            axum::response::Html(include_str!("../web/index.html"))
        }))
        .route("/health", get(|| async { "OK" }))
        .nest("/api", api)
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)); // 100MB limit for large uploads

    let listener = match tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            eprintln!("\n\x1b[31mError:\x1b[0m Port {} is already in use.\n", port);
            eprintln!("Try a different port with:");
            eprintln!("  \x1b[36meywa serve --port <PORT>\x1b[0m\n");
            eprintln!("Example:");
            eprintln!("  eywa serve --port 8006");
            return Err(e.into());
        }
        Err(e) => return Err(e.into()),
    };
    println!("Server running on http://localhost:{}", port);
    println!("Web UI v1:       http://localhost:{}/v1", port);
    println!("\nAPI Endpoints:");
    println!("  GET    /health                  - Health check");
    println!("  GET    /api/info                - System info (models, storage, stats)");
    println!("  POST   /api/search              - Search documents");
    println!("  POST   /api/ingest              - Add documents (sync/blocking)");
    println!("  POST   /api/ingest/async        - Add documents (async/background)");
    println!("  GET    /api/jobs                - List all jobs");
    println!("  GET    /api/jobs/:id            - Get job progress");
    println!("  GET    /api/jobs/:id/docs       - Get per-document status");
    println!("  GET    /api/sources             - List all sources");
    println!("  DELETE /api/sources/:id         - Delete a source");
    println!("  GET    /api/sources/:id/docs    - List documents in source");
    println!("  GET    /api/sources/:id/export  - Export source as zip");
    println!("  GET    /api/docs/:id            - Get document content");
    println!("  DELETE /api/docs/:id            - Delete a document");
    println!("  GET    /api/export              - Export all docs as zip");
    println!("  DELETE /api/reset               - Reset all data");
    println!("\nBackground worker started (jobs persist across restarts).");

    axum::serve(listener, app).await?;
    Ok(())
}

/// Background worker that processes the job queue
/// Processes docs individually for granular status tracking
async fn run_queue_worker(
    job_queue: SharedJobQueue,
    embedder: std::sync::Arc<Embedder>,
    db: std::sync::Arc<tokio::sync::RwLock<VectorDB>>,
    bm25_index: std::sync::Arc<BM25Index>,
    data_dir: String,
) {
    let mut cleanup_counter = 0u32;

    loop {
        // Get next pending doc (already marked as processing by get_next_pending)
        let doc_result = {
            let mut queue = job_queue.lock().unwrap();
            queue.get_next_pending()
        };

        let doc = match doc_result {
            Ok(Some(d)) => d,
            Ok(None) => {
                // No work, sleep a bit
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                cleanup_counter += 1;
                if cleanup_counter >= 100 {
                    cleanup_counter = 0;
                    let mut queue = job_queue.lock().unwrap();
                    if let Err(e) = queue.cleanup_old_jobs(3600) {
                        eprintln!("Error cleaning up old jobs: {}", e);
                    }
                }
                continue;
            }
            Err(e) => {
                eprintln!("Worker error getting doc: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                continue;
            }
        };

        // Process single document
        let doc_id = doc.id.clone();
        let result = process_single_document(&embedder, &db, &bm25_index, &data_dir, &doc).await;

        // Mark completed or failed
        let mut queue = job_queue.lock().unwrap();
        match result {
            Ok(_) => {
                if let Err(e) = queue.mark_completed(&doc_id) {
                    eprintln!("Error marking doc {} completed: {}", doc_id, e);
                }
            }
            Err(e) => {
                if let Err(err) = queue.mark_failed(&doc_id, &e.to_string()) {
                    eprintln!("Error marking doc {} failed: {}", doc_id, err);
                }
            }
        }

        // Reset cleanup counter when we're doing work
        cleanup_counter = 0;
    }
}

/// Process a batch of documents together for efficient embedding
/// Key: Embedding happens OUTSIDE the DB lock to avoid blocking reads
async fn process_document_batch(
    embedder: &std::sync::Arc<Embedder>,
    db_lock: &std::sync::Arc<tokio::sync::RwLock<VectorDB>>,
    bm25_index: &std::sync::Arc<BM25Index>,
    data_dir: &str,
    docs: &[eywa::PendingDoc],
) -> Result<()> {
    use std::collections::HashMap;

    // Group by source_id for proper source attribution
    let mut by_source: HashMap<String, Vec<DocumentInput>> = HashMap::new();
    for doc in docs {
        by_source
            .entry(doc.source_id.clone())
            .or_default()
            .push(DocumentInput {
                content: doc.content.clone(),
                title: doc.title.clone(),
                file_path: doc.file_path.clone(),
            });
    }

    let pipeline = IngestPipeline::new(std::sync::Arc::clone(embedder), std::sync::Arc::clone(bm25_index));
    let data_path = std::path::Path::new(data_dir);

    // Process each source: embed OUTSIDE lock, then write WITH lock
    for (source_id, inputs) in by_source {
        // Step 1: Prepare + embed (slow) - NO LOCK HELD
        let embedded_batch = pipeline.prepare_and_embed(&source_id, data_path, inputs)?;

        // Step 2: Write to DB (fast) - lock held briefly
        {
            let mut db = db_lock.write().await;
            pipeline.write_embedded_batch(&mut db, embedded_batch).await?;
        } // Lock released immediately
    }
    Ok(())
}

/// Process a single document for granular status tracking
/// Embedding efficiency is maintained at the chunk level
async fn process_single_document(
    embedder: &std::sync::Arc<Embedder>,
    db_lock: &std::sync::Arc<tokio::sync::RwLock<VectorDB>>,
    bm25_index: &std::sync::Arc<BM25Index>,
    data_dir: &str,
    doc: &eywa::PendingDoc,
) -> Result<()> {
    let pipeline = IngestPipeline::new(std::sync::Arc::clone(embedder), std::sync::Arc::clone(bm25_index));
    let data_path = std::path::Path::new(data_dir);

    let input = DocumentInput {
        content: doc.content.clone(),
        title: doc.title.clone(),
        file_path: doc.file_path.clone(),
    };

    // Step 1: Prepare + embed (slow) - NO LOCK HELD
    let embedded_batch = pipeline.prepare_and_embed(&doc.source_id, data_path, vec![input])?;

    // Step 2: Write to DB (fast) - lock held briefly
    {
        let mut db = db_lock.write().await;
        pipeline.write_embedded_batch(&mut db, embedded_batch).await?;
    }

    Ok(())
}

/// Run the MCP server (JSON-RPC over stdio)
async fn run_mcp_server(data_dir: &str) -> Result<()> {
    use serde_json::{json, Value};
    use std::io::{BufRead, BufReader, Write};

    let embedder = Embedder::new()?;
    let db = VectorDB::new(data_dir).await?;
    let content_store = ContentStore::open(&std::path::Path::new(data_dir).join("content.db"))?;
    let search_engine = SearchEngine::with_reranker()?;

    let stdin = std::io::stdin();
    let reader = BufReader::new(stdin.lock());
    let mut stdout = std::io::stdout();

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let error = json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": -32700, "message": format!("Parse error: {}", e) }
                });
                writeln!(stdout, "{}", error)?;
                stdout.flush()?;
                continue;
            }
        };

        let id = request.get("id").cloned();
        let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");

        let response = match method {
            "initialize" => {
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "eywa",
                            "version": "0.1.0"
                        }
                    }
                })
            }

            "notifications/initialized" | "initialized" => {
                continue; // No response needed for notifications
            }

            "tools/list" => {
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": [
                            {
                                "name": "search",
                                "description": "Search the knowledge base for relevant documents. Uses hybrid vector + keyword search with neural reranking.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "The search query"
                                        },
                                        "limit": {
                                            "type": "integer",
                                            "description": "Maximum number of results (default: 5)",
                                            "default": 5
                                        },
                                        "source": {
                                            "type": "string",
                                            "description": "Optional: filter results to a specific source"
                                        }
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "similar_docs",
                                "description": "Find documents similar to a given document. Returns reranked results.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "document_id": {
                                            "type": "string",
                                            "description": "The document ID to find similar documents for"
                                        },
                                        "limit": {
                                            "type": "integer",
                                            "description": "Maximum number of results (default: 5)",
                                            "default": 5
                                        }
                                    },
                                    "required": ["document_id"]
                                }
                            },
                            {
                                "name": "list_sources",
                                "description": "List all document sources in the knowledge base",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {}
                                }
                            },
                            {
                                "name": "list_documents",
                                "description": "List all documents in a specific source. Returns document titles, file paths, and IDs.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "source_id": {
                                            "type": "string",
                                            "description": "The source ID to list documents from"
                                        }
                                    },
                                    "required": ["source_id"]
                                }
                            },
                            {
                                "name": "get_document",
                                "description": "Get the full content of a specific document by ID",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "document_id": {
                                            "type": "string",
                                            "description": "The document ID to retrieve"
                                        }
                                    },
                                    "required": ["document_id"]
                                }
                            }
                        ]
                    }
                })
            }

            "tools/call" => {
                let params = request.get("params").cloned().unwrap_or(json!({}));
                let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

                match tool_name {
                    "search" => {
                        let query = arguments.get("query").and_then(|q| q.as_str()).unwrap_or("");
                        let limit = arguments.get("limit").and_then(|l| l.as_u64()).unwrap_or(5) as usize;
                        let source = arguments.get("source").and_then(|s| s.as_str());

                        match embedder.embed(query) {
                            Ok(embedding) => {
                                match db.search_filtered(&embedding, limit * 2, source).await {
                                    Ok(chunk_metas) => {
                                        // Fetch content from SQLite
                                        let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
                                        let contents = match content_store.get_chunks(&chunk_ids) {
                                            Ok(c) => c,
                                            Err(e) => {
                                                let resp = json!({
                                                    "jsonrpc": "2.0",
                                                    "id": id,
                                                    "error": { "code": -32000, "message": format!("Content fetch error: {}", e) }
                                                });
                                                writeln!(stdout, "{}", resp)?;
                                                stdout.flush()?;
                                                continue;
                                            }
                                        };
                                        let content_map: std::collections::HashMap<String, String> = contents.into_iter().collect();

                                        // Combine metadata + content
                                        let results: Vec<SearchResult> = chunk_metas
                                            .into_iter()
                                            .filter_map(|meta| {
                                                let content = content_map.get(&meta.id)?.clone();
                                                Some(SearchResult {
                                                    id: meta.id,
                                                    source_id: meta.source_id,
                                                    title: meta.title,
                                                    content,
                                                    file_path: meta.file_path,
                                                    line_start: meta.line_start,
                                                    score: meta.score,
                                                })
                                            })
                                            .collect();

                                        let results = search_engine.filter_results(results);
                                        let results = search_engine.rerank(results, query, limit);

                                        let text = results.iter().map(|r| {
                                            format!(
                                                "## {} (Score: {:.3})\nSource: {}\n\n{}",
                                                r.title.as_deref().unwrap_or("Untitled"),
                                                r.score,
                                                r.source_id,
                                                r.content
                                            )
                                        }).collect::<Vec<_>>().join("\n\n---\n\n");

                                        json!({
                                            "jsonrpc": "2.0",
                                            "id": id,
                                            "result": {
                                                "content": [{
                                                    "type": "text",
                                                    "text": if results.is_empty() {
                                                        "No results found.".to_string()
                                                    } else {
                                                        format!("Found {} results:\n\n{}", results.len(), text)
                                                    }
                                                }]
                                            }
                                        })
                                    }
                                    Err(e) => json!({
                                        "jsonrpc": "2.0",
                                        "id": id,
                                        "error": { "code": -32000, "message": format!("Search error: {}", e) }
                                    })
                                }
                            }
                            Err(e) => json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": { "code": -32000, "message": format!("Embedding error: {}", e) }
                            })
                        }
                    }

                    "list_sources" => {
                        match db.list_sources().await {
                            Ok(sources) => {
                                let text = if sources.is_empty() {
                                    "No sources found in the knowledge base.".to_string()
                                } else {
                                    sources.iter().map(|s| {
                                        format!("- {} ({} chunks)", s.name, s.chunk_count)
                                    }).collect::<Vec<_>>().join("\n")
                                };

                                json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "result": {
                                        "content": [{
                                            "type": "text",
                                            "text": format!("Sources:\n{}", text)
                                        }]
                                    }
                                })
                            }
                            Err(e) => json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": { "code": -32000, "message": format!("Error: {}", e) }
                            })
                        }
                    }

                    "list_documents" => {
                        let source_id = arguments.get("source_id").and_then(|s| s.as_str()).unwrap_or("");
                        if source_id.is_empty() {
                            json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": { "code": -32602, "message": "source_id is required" }
                            })
                        } else {
                            match db.list_documents(source_id, Some(db::MAX_QUERY_LIMIT)).await {
                                Ok(docs) => {
                                    let text = if docs.is_empty() {
                                        format!("No documents found in source '{}'.", source_id)
                                    } else {
                                        docs.iter().map(|d| {
                                            let file_info = d.file_path.as_ref()
                                                .map(|p| format!(" ({})", p))
                                                .unwrap_or_default();
                                            format!("- [{}] {}{} - {} chunks, {} chars",
                                                d.id, d.title, file_info, d.chunk_count, d.content_length)
                                        }).collect::<Vec<_>>().join("\n")
                                    };

                                    json!({
                                        "jsonrpc": "2.0",
                                        "id": id,
                                        "result": {
                                            "content": [{
                                                "type": "text",
                                                "text": format!("Documents in '{}':\n{}", source_id, text)
                                            }]
                                        }
                                    })
                                }
                                Err(e) => json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "error": { "code": -32000, "message": format!("Error: {}", e) }
                                })
                            }
                        }
                    }

                    "get_document" => {
                        let doc_id = arguments.get("document_id").and_then(|s| s.as_str()).unwrap_or("");
                        if doc_id.is_empty() {
                            json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": { "code": -32602, "message": "document_id is required" }
                            })
                        } else {
                            match db.get_document(doc_id).await {
                                Ok(Some(record)) => {
                                    // Fetch content from SQLite
                                    let content = match content_store.get_document(doc_id) {
                                        Ok(Some(c)) => c,
                                        Ok(None) => {
                                            let resp = json!({
                                                "jsonrpc": "2.0",
                                                "id": id,
                                                "error": { "code": -32000, "message": format!("Document content not found: {}", doc_id) }
                                            });
                                            writeln!(stdout, "{}", resp)?;
                                            stdout.flush()?;
                                            continue;
                                        }
                                        Err(e) => {
                                            let resp = json!({
                                                "jsonrpc": "2.0",
                                                "id": id,
                                                "error": { "code": -32000, "message": format!("Content fetch error: {}", e) }
                                            });
                                            writeln!(stdout, "{}", resp)?;
                                            stdout.flush()?;
                                            continue;
                                        }
                                    };

                                    let file_info = record.file_path.as_ref()
                                        .map(|p| format!("\nFile: {}", p))
                                        .unwrap_or_default();
                                    json!({
                                        "jsonrpc": "2.0",
                                        "id": id,
                                        "result": {
                                            "content": [{
                                                "type": "text",
                                                "text": format!(
                                                    "# {}\nSource: {}{}\nCreated: {}\n\n{}",
                                                    record.title, record.source_id, file_info, record.created_at, content
                                                )
                                            }]
                                        }
                                    })
                                }
                                Ok(None) => json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "error": { "code": -32000, "message": format!("Document not found: {}", doc_id) }
                                }),
                                Err(e) => json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "error": { "code": -32000, "message": format!("Error: {}", e) }
                                })
                            }
                        }
                    }

                    "similar_docs" => {
                        let doc_id = arguments.get("document_id").and_then(|s| s.as_str()).unwrap_or("");
                        let limit = arguments.get("limit").and_then(|l| l.as_u64()).unwrap_or(5) as usize;

                        if doc_id.is_empty() {
                            json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": { "code": -32602, "message": "document_id is required" }
                            })
                        } else {
                            // Get source document content
                            let source_content = match content_store.get_document(doc_id) {
                                Ok(Some(c)) => c,
                                Ok(None) => {
                                    let resp = json!({
                                        "jsonrpc": "2.0",
                                        "id": id,
                                        "error": { "code": -32000, "message": format!("Document not found: {}", doc_id) }
                                    });
                                    writeln!(stdout, "{}", resp)?;
                                    stdout.flush()?;
                                    continue;
                                }
                                Err(e) => {
                                    let resp = json!({
                                        "jsonrpc": "2.0",
                                        "id": id,
                                        "error": { "code": -32000, "message": format!("Error fetching document: {}", e) }
                                    });
                                    writeln!(stdout, "{}", resp)?;
                                    stdout.flush()?;
                                    continue;
                                }
                            };

                            // Embed the source document
                            match embedder.embed(&source_content) {
                                Ok(embedding) => {
                                    // Search for similar chunks (get more to filter out self)
                                    match db.search(&embedding, (limit + 5) * 2).await {
                                        Ok(chunk_metas) => {
                                            // Filter out chunks from the same document
                                            let chunk_metas: Vec<_> = chunk_metas
                                                .into_iter()
                                                .filter(|c| c.document_id != doc_id)
                                                .collect();

                                            // Fetch content from SQLite
                                            let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
                                            let contents = match content_store.get_chunks(&chunk_ids) {
                                                Ok(c) => c,
                                                Err(e) => {
                                                    let resp = json!({
                                                        "jsonrpc": "2.0",
                                                        "id": id,
                                                        "error": { "code": -32000, "message": format!("Content fetch error: {}", e) }
                                                    });
                                                    writeln!(stdout, "{}", resp)?;
                                                    stdout.flush()?;
                                                    continue;
                                                }
                                            };
                                            let content_map: std::collections::HashMap<String, String> = contents.into_iter().collect();

                                            // Combine metadata + content
                                            let results: Vec<SearchResult> = chunk_metas
                                                .into_iter()
                                                .filter_map(|meta| {
                                                    let content = content_map.get(&meta.id)?.clone();
                                                    Some(SearchResult {
                                                        id: meta.id,
                                                        source_id: meta.source_id,
                                                        title: meta.title,
                                                        content,
                                                        file_path: meta.file_path,
                                                        line_start: meta.line_start,
                                                        score: meta.score,
                                                    })
                                                })
                                                .collect();

                                            // Rerank against source document content
                                            let results = search_engine.rerank(results, &source_content, limit);

                                            let text = results.iter().map(|r| {
                                                format!(
                                                    "## {} (Score: {:.3})\nSource: {}\n\n{}",
                                                    r.title.as_deref().unwrap_or("Untitled"),
                                                    r.score,
                                                    r.source_id,
                                                    r.content
                                                )
                                            }).collect::<Vec<_>>().join("\n\n---\n\n");

                                            json!({
                                                "jsonrpc": "2.0",
                                                "id": id,
                                                "result": {
                                                    "content": [{
                                                        "type": "text",
                                                        "text": if results.is_empty() {
                                                            "No similar documents found.".to_string()
                                                        } else {
                                                            format!("Found {} similar documents:\n\n{}", results.len(), text)
                                                        }
                                                    }]
                                                }
                                            })
                                        }
                                        Err(e) => json!({
                                            "jsonrpc": "2.0",
                                            "id": id,
                                            "error": { "code": -32000, "message": format!("Search error: {}", e) }
                                        })
                                    }
                                }
                                Err(e) => json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "error": { "code": -32000, "message": format!("Embedding error: {}", e) }
                                })
                            }
                        }
                    }

                    _ => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": format!("Unknown tool: {}", tool_name) }
                    })
                }
            }

            _ => {
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": format!("Method not found: {}", method) }
                })
            }
        };

        writeln!(stdout, "{}", response)?;
        stdout.flush()?;
    }

    Ok(())
}

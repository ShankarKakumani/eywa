//! Init command handler

use anyhow::Result;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use eywa::{
    run_download_wizard, run_init, BM25Index, Config, ContentStore,
    DocumentInput, Embedder, IngestPipeline, InitResult, Reranker, VectorDB,
};

pub async fn run_init_command(data_dir: &str, default: bool) -> Result<()> {
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
    let marker_path = Path::new(data_dir).join(".reindex_in_progress");
    let interrupted = marker_path.exists();

    match run_init(existing.as_ref()).await? {
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
                let content_path = Path::new(data_dir).join("content.db");
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
                    let mut db = VectorDB::new(data_dir).await?;
                    db.reset_all().await?;
                    let data_path = Path::new(data_dir);
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
                            is_pdf: false,
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

    Ok(())
}

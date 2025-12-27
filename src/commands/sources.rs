//! Sources, docs, and delete command handlers

use anyhow::Result;
use std::path::Path;
use eywa::{db, BM25Index, ContentStore, VectorDB};

pub async fn run_sources(data_dir: &str) -> Result<()> {
    let db = VectorDB::new(data_dir).await?;
    let sources = db.list_sources().await?;

    if sources.is_empty() {
        println!("No sources found. Use 'eywa ingest' to add documents.");
    } else {
        println!("Sources:\n");
        for source in sources {
            println!("  {} ({} chunks)", source.name, source.chunk_count);
        }
    }

    Ok(())
}

pub async fn run_docs(data_dir: &str, source: &str) -> Result<()> {
    let db = VectorDB::new(data_dir).await?;
    let docs = db.list_documents(source, Some(db::MAX_QUERY_LIMIT)).await?;

    if docs.is_empty() {
        println!("No documents found in source '{}'.", source);
    } else {
        println!("Documents in '{}':\n", source);
        for doc in docs {
            println!("  {} - {} ({} chunks, {} chars)",
                doc.id, doc.title, doc.chunk_count, doc.content_length);
        }
    }

    Ok(())
}

pub async fn run_delete(data_dir: &str, source: &str) -> Result<()> {
    let data_path = Path::new(data_dir);
    let db = VectorDB::new(data_dir).await?;
    let bm25_index = BM25Index::open(data_path)?;
    let content_store = ContentStore::open(&data_path.join("content.db"))?;

    // Get document IDs for SQLite cleanup
    let doc_ids = db.get_document_ids_for_source(source).await?;
    let doc_id_refs: Vec<&str> = doc_ids.iter().map(|s| s.as_str()).collect();

    // Delete from all stores
    db.delete_source(source).await?;
    bm25_index.delete_source(source)?;
    content_store.delete_source(&doc_id_refs)?;

    println!("Deleted source: {}", source);

    Ok(())
}

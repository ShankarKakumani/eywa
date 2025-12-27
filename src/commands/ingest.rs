//! Ingest command handler

use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use eywa::{BM25Index, Embedder, IngestPipeline, VectorDB};

pub async fn run_ingest(data_dir: &str, source: &str, path: &Path) -> Result<()> {
    println!("Initializing embedder...");
    let embedder = Arc::new(Embedder::new()?);

    println!("Connecting to database...");
    let mut db = VectorDB::new(data_dir).await?;
    let data_path = Path::new(data_dir);
    let bm25_index = Arc::new(BM25Index::open(data_path)?);

    println!("Ingesting documents from: {}\n", path.display());
    let pipeline = IngestPipeline::new(embedder, bm25_index);

    let path_str = path.to_string_lossy().to_string();
    let result = pipeline.ingest_from_path(&mut db, data_path, source, &path_str).await?;

    println!("\nIngestion complete!");
    println!("  Source: {}", result.source_id);
    println!("  Documents created: {}", result.documents_created);
    println!("  Chunks created: {}", result.chunks_created);
    println!("  Chunks skipped (duplicates): {}", result.chunks_skipped);

    Ok(())
}

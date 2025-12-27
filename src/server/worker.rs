//! Background queue worker for async document processing

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use eywa::{
    BM25Index, DocumentInput, Embedder, IngestPipeline,
    PendingDoc, SharedJobQueue, VectorDB,
};

/// Background worker that processes the job queue
/// Processes docs individually for granular status tracking
pub async fn run_queue_worker(
    job_queue: SharedJobQueue,
    embedder: Arc<Embedder>,
    db: Arc<RwLock<VectorDB>>,
    bm25_index: Arc<BM25Index>,
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

/// Process a single document from the queue
async fn process_single_document(
    embedder: &Arc<Embedder>,
    db_lock: &Arc<RwLock<VectorDB>>,
    bm25_index: &Arc<BM25Index>,
    data_dir: &str,
    doc: &PendingDoc,
) -> Result<()> {
    let pipeline = IngestPipeline::new(Arc::clone(embedder), Arc::clone(bm25_index));
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

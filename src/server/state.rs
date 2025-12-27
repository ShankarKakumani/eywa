//! Server application state

use std::sync::Arc;
use tokio::sync::RwLock;
use eywa::{BM25Index, Embedder, SearchEngine, SharedJobQueue, VectorDB};

/// Shared application state for all route handlers
pub struct AppState {
    pub embedder: Arc<Embedder>,
    pub db: Arc<RwLock<VectorDB>>,
    pub bm25_index: Arc<BM25Index>,
    pub search_engine: SearchEngine,
    pub job_queue: SharedJobQueue,
    pub data_dir: String,
}

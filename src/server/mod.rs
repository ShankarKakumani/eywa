//! HTTP server module

mod state;
mod routes;
mod worker;

pub use state::AppState;
use routes::create_router;
pub use worker::run_queue_worker;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use eywa::{create_job_queue, BM25Index, Embedder, SearchEngine, VectorDB};

/// Run the HTTP server
pub async fn run_server(data_dir: &str, port: u16) -> Result<()> {
    // Shared components
    let embedder = Arc::new(Embedder::new()?);
    let db = Arc::new(RwLock::new(VectorDB::new(data_dir).await?));
    let bm25_index = Arc::new(BM25Index::open(std::path::Path::new(data_dir))?);
    let search_engine = SearchEngine::new();
    let job_db_path = std::path::Path::new(data_dir).join("jobs.db");
    let job_queue = create_job_queue(&job_db_path)?;

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

    // Create router
    let app = create_router(state);

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

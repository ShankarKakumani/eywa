//! HTTP API route handlers

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json, Response},
    routing::{delete, get, patch, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use eywa::{db, chunking, Config, ContentStore, DevicePreference, DocumentInput, EmbeddingModelConfig, FetchUrlRequest, gpu_support_info, IngestPipeline, IngestRequest, RerankerModelConfig, SearchRequest, SearchResult, LLMConfig};
use eywa::setup::{DownloadProgress, ModelDownloader, ModelInfo};
use eywa::llm::{Message, create_provider};
use crate::server::{AppState, DownloadJob, DownloadStatus, DownloadTracker, FileProgress};
use crate::utils::{create_zip, dir_size, extract_text_from_html, extract_title_from_html, lance_db_size, scan_hf_cache};

/// Capitalize device name to match available_devices format (Auto, Cpu, Metal, Cuda)
fn capitalize_device(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Preprocess documents: extract text from PDFs before queuing
fn preprocess_documents(documents: Vec<DocumentInput>) -> Vec<DocumentInput> {
    documents.into_iter().filter_map(|doc| {
        if doc.is_pdf {
            // Extract text from base64 PDF
            match chunking::extract_text_from_base64_pdf(&doc.content) {
                Ok(text) => Some(DocumentInput {
                    content: text,
                    title: doc.title,
                    file_path: doc.file_path,
                    is_pdf: false, // Now it's extracted text
                }),
                Err(e) => {
                    eprintln!("Warning: Failed to extract PDF {}: {}",
                        doc.title.as_deref().unwrap_or("untitled"), e);
                    None
                }
            }
        } else {
            Some(doc)
        }
    }).collect()
}

/// Create the main application router
pub fn create_router(state: Arc<AppState>) -> Router {
    let api = create_api_routes(state);

    Router::new()
        // Web UI v2 (default)
        .route("/", get(|| async {
            Html(include_str!("../../web/v2/index.html"))
        }))
        .route("/style.css", get(|| async {
            (
                [(header::CONTENT_TYPE, "text/css")],
                include_str!("../../web/v2/style.css")
            )
        }))
        .route("/api.js", get(|| async {
            (
                [(header::CONTENT_TYPE, "application/javascript")],
                include_str!("../../web/v2/api.js")
            )
        }))
        .route("/app.js", get(|| async {
            (
                [(header::CONTENT_TYPE, "application/javascript")],
                include_str!("../../web/v2/app.js")
            )
        }))
        .route("/dashboard.js", get(|| async {
            (
                [(header::CONTENT_TYPE, "application/javascript")],
                include_str!("../../web/v2/dashboard.js")
            )
        }))
        .route("/add-docs.js", get(|| async {
            (
                [(header::CONTENT_TYPE, "application/javascript")],
                include_str!("../../web/v2/add-docs.js")
            )
        }))
        .route("/explorer.js", get(|| async {
            (
                [(header::CONTENT_TYPE, "application/javascript")],
                include_str!("../../web/v2/explorer.js")
            )
        }))
        .route("/jobs.js", get(|| async {
            (
                [(header::CONTENT_TYPE, "application/javascript")],
                include_str!("../../web/v2/jobs.js")
            )
        }))
        .route("/settings.js", get(|| async {
            (
                [(header::CONTENT_TYPE, "application/javascript")],
                include_str!("../../web/v2/settings.js")
            )
        }))
        .route("/favicon.png", get(|| async {
            (
                [(header::CONTENT_TYPE, "image/png")],
                include_bytes!("../../web/v2/favicon.png").as_slice()
            )
        }))
        .route("/apple-touch-icon.png", get(|| async {
            (
                [(header::CONTENT_TYPE, "image/png")],
                include_bytes!("../../web/v2/apple-touch-icon.png").as_slice()
            )
        }))
        // Web UI v1 (legacy)
        .route("/v1", get(|| async {
            Html(include_str!("../../web/index.html"))
        }))
        .route("/health", get(|| async { "OK" }))
        .nest("/api", api)
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // 100MB limit
}

/// Create API routes
fn create_api_routes(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/info", get(handle_info))
        .route("/search", post(handle_search))
        .route("/ingest", post(handle_ingest))
        .route("/queue", post(handle_queue))
        .route("/ingest/async", post(handle_ingest_async))
        .route("/jobs", get(handle_list_jobs))
        .route("/jobs/:job_id", get(handle_get_job))
        .route("/jobs/:job_id/docs", get(handle_get_job_docs))
        .route("/sources", get(handle_list_sources))
        .route("/sources/:source_id", delete(handle_delete_source))
        .route("/sources/:source_id/docs", get(handle_list_source_docs))
        .route("/sources/:source_id/export", get(handle_export_source))
        .route("/docs/:doc_id", get(handle_get_doc))
        .route("/docs/:doc_id", delete(handle_delete_doc))
        .route("/sql/sources", get(handle_sql_sources))
        .route("/sql/sources/:source_id/docs", get(handle_sql_source_docs))
        .route("/reset", delete(handle_reset))
        .route("/export", get(handle_export))
        .route("/fetch-preview", post(handle_fetch_preview))
        .route("/fetch-url", post(handle_fetch_url))
        // Settings & Models API
        .route("/settings", get(handle_get_settings))
        .route("/settings", patch(handle_update_settings))
        .route("/models/embedders", get(handle_list_embedders))
        .route("/models/rerankers", get(handle_list_rerankers))
        // Model Download API
        .route("/models/download", post(handle_start_download))
        .route("/models/download/:job_id", get(handle_get_download))
        .route("/models/downloads", get(handle_list_downloads))
        .route("/models/cache/:model_type/:model_id", delete(handle_delete_model_cache))
        // Chat API
        .route("/chat", post(handle_chat))
        .with_state(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Route Handlers
// ─────────────────────────────────────────────────────────────────────────────

async fn handle_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = Config::load().ok().flatten();
    let db = state.db.read().await;
    let sources = db.list_sources().await.unwrap_or_default();

    let source_count = sources.len();
    let chunk_count: u64 = sources.iter().map(|s| s.chunk_count).sum();

    let document_count = ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db"))
        .ok()
        .and_then(|cs| cs.count_documents().ok())
        .unwrap_or(0);

    let data_path = std::path::Path::new(&state.data_dir);
    let content_db_bytes = std::fs::metadata(data_path.join("content.db"))
        .map(|m| m.len())
        .unwrap_or(0);
    let vector_db_bytes = lance_db_size(data_path);
    let bm25_index_bytes = dir_size(&data_path.join("tantivy")).unwrap_or(0);

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

    if let Some(cfg) = config {
        response["embedding_model"] = json!({
            "name": cfg.embedding_model.name,
            "size_mb": cfg.embedding_model.size_mb,
            "dimensions": cfg.embedding_model.dimensions
        });
        response["reranker_model"] = json!({
            "name": cfg.reranker_model.name,
            "size_mb": cfg.reranker_model.size_mb
        });
    }

    let cached_models = scan_hf_cache();
    let cached_models_json: Vec<_> = cached_models.iter().map(|m| {
        json!({
            "name": m.name,
            "size_bytes": m.size_bytes
        })
    }).collect();
    response["cached_models"] = json!(cached_models_json);

    // Add app info (version, GPU support)
    let gpu_info = gpu_support_info();
    response["app"] = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "gpu_support": gpu_info.summary()
    });

    (StatusCode::OK, Json(response))
}

async fn handle_search(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>,
) -> impl IntoResponse {
    let query_embedding = match state.embedder.embed(&payload.query) {
        Ok(e) => e,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    let db = state.db.read().await;
    let chunk_metas = match db.search(&query_embedding, payload.limit * 2).await {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    drop(db); // Release lock

    // ... existing search logic ...

    // TODO: Refactor search logic to use Eywa::search if possible, but for now this works. 
    // Wait, the handle_search implementation above seems cut off in the view. 
    // I should simply append handle_chat at the end of the file or after handle_search.
    // I will append it after handle_search instead of replacing inside it.
    
    // Actually, I'll just append handle_chat before handle_list_embedders or similar.
    // Let's scroll down to find a good spot.
    
    // Wait, I can't see the end of handle_search. 
    // I'll put handle_chat AFTER handle_update_settings (which is around line 880 in previous view).
    // Or I can add it at the very end of route handlers section.
    
    // Let's add it before handle_list_embedders (line 884).
    
    // I will use a separate replacement chunk for that.
    // Placeholder removed to allow flow to continue
    
    // SKIPPING THIS CHUNK as I will add handle_chat elsewhere.
    // SKIPPING THIS CHUNK as I will add handle_chat elsewhere.

    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
        Ok(cs) => cs,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
    let contents = match content_store.get_chunks(&chunk_ids) {
        Ok(c) => c,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };
    let content_map: HashMap<String, String> = contents.into_iter().collect();

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

    (StatusCode::OK, Json(json!({
        "query": payload.query,
        "results": results,
        "count": count
    })))
}

async fn handle_ingest(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IngestRequest>,
) -> impl IntoResponse {
    let data_dir = std::path::Path::new(&state.data_dir);
    let mut db = state.db.write().await;
    let pipeline = IngestPipeline::new(Arc::clone(&state.embedder), Arc::clone(&state.bm25_index));

    match pipeline.ingest_documents(&mut db, data_dir, &payload.source_id, payload.documents).await {
        Ok(result) => (StatusCode::OK, Json(json!(result))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    }
}

async fn handle_queue(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IngestRequest>,
) -> impl IntoResponse {
    // Preprocess PDFs: extract text from base64 content
    let documents = preprocess_documents(payload.documents);

    let result = {
        let mut queue = state.job_queue.lock().unwrap();
        queue.queue_documents(&payload.source_id, documents.clone())
    };
    match result {
        Ok(job_id) => {
            let docs_queued = documents.len() as u32;
            (StatusCode::ACCEPTED, Json(json!({
                "job_id": job_id,
                "docs_queued": docs_queued,
                "message": format!("Queued {} documents for processing", docs_queued)
            })))
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))
    }
}

async fn handle_ingest_async(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IngestRequest>,
) -> impl IntoResponse {
    // Preprocess PDFs: extract text from base64 content
    let documents = preprocess_documents(payload.documents);

    let result = {
        let mut queue = state.job_queue.lock().unwrap();
        queue.queue_documents(&payload.source_id, documents.clone())
    };
    match result {
        Ok(job_id) => {
            let total_docs = documents.len() as u32;
            (StatusCode::ACCEPTED, Json(json!({
                "job_id": job_id,
                "status": "queued",
                "total_docs": total_docs
            })))
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))
    }
}

async fn handle_list_jobs(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let result = {
        let queue = state.job_queue.lock().unwrap();
        queue.list_jobs()
    };
    match result {
        Ok(jobs) => (StatusCode::OK, Json(json!({ "jobs": jobs }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))
    }
}

async fn handle_get_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
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

async fn handle_get_job_docs(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let result = {
        let queue = state.job_queue.lock().unwrap();
        queue.get_job_docs(&job_id)
    };
    match result {
        Ok(docs) => (StatusCode::OK, Json(json!({ "docs": docs }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))
    }
}

async fn handle_list_sources(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    match db.list_sources().await {
        Ok(sources) => (StatusCode::OK, Json(json!({ "sources": sources }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    }
}

async fn handle_delete_source(
    State(state): State<Arc<AppState>>,
    Path(source_id): Path<String>,
) -> impl IntoResponse {
    let db = state.db.read().await;

    if let Err(e) = db.delete_source(&source_id).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })));
    }

    if let Err(e) = state.bm25_index.delete_source(&source_id) {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })));
    }

    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
        Ok(cs) => cs,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    if let Err(e) = content_store.delete_source_by_source_id(&source_id) {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })));
    }

    (StatusCode::OK, Json(json!({ "deleted": source_id })))
}

async fn handle_list_source_docs(
    State(state): State<Arc<AppState>>,
    Path(source_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let limit = params.get("limit").and_then(|v| {
        if v == "all" { Some(db::MAX_QUERY_LIMIT) } else { v.parse().ok() }
    });

    let db = state.db.read().await;
    match db.list_documents(&source_id, limit).await {
        Ok(docs) => (StatusCode::OK, Json(json!({ "documents": docs }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    }
}

async fn handle_get_doc(
    State(state): State<Arc<AppState>>,
    Path(doc_id): Path<String>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let record = match db.get_document(&doc_id).await {
        Ok(Some(r)) => r,
        Ok(None) => return (StatusCode::NOT_FOUND, Json(json!({ "error": "Document not found" }))),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
        Ok(cs) => cs,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    let content = match content_store.get_document(&doc_id) {
        Ok(Some(c)) => c,
        Ok(None) => return (StatusCode::NOT_FOUND, Json(json!({ "error": "Document content not found" }))),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

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

async fn handle_delete_doc(
    State(state): State<Arc<AppState>>,
    Path(doc_id): Path<String>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    if let Err(e) = db.delete_document(&doc_id).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })));
    }

    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
        Ok(cs) => cs,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    if let Err(e) = content_store.delete_document(&doc_id) {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })));
    }

    (StatusCode::OK, Json(json!({ "deleted": doc_id })))
}

async fn handle_sql_sources(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
        Ok(cs) => cs,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    match content_store.list_sources() {
        Ok(sources) => (StatusCode::OK, Json(json!({ "sources": sources }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    }
}

async fn handle_sql_source_docs(
    State(state): State<Arc<AppState>>,
    Path(source_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let limit = params.get("limit").and_then(|v| {
        if v == "all" { None } else { v.parse().ok() }
    });
    let offset = params.get("offset").and_then(|v| v.parse().ok());

    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
        Ok(cs) => cs,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    match content_store.list_documents_by_source(&source_id, limit, offset) {
        Ok((docs, total)) => (StatusCode::OK, Json(json!({
            "documents": docs,
            "total_documents": total
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    }
}

async fn handle_reset(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut db = state.db.write().await;
    if let Err(e) = db.reset_all().await {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })));
    }

    if let Err(e) = state.bm25_index.reset() {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })));
    }

    let content_store = match ContentStore::open(&std::path::Path::new(&state.data_dir).join("content.db")) {
        Ok(cs) => cs,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    if let Err(e) = content_store.reset() {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })));
    }

    (StatusCode::OK, Json(json!({ "status": "reset complete" })))
}

async fn handle_export(State(state): State<Arc<AppState>>) -> Response {
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

    let docs: Vec<eywa::Document> = doc_rows
        .into_iter()
        .map(|r| eywa::Document {
            id: r.id,
            source_id: r.source_id,
            title: r.title,
            content: r.content,
            file_path: r.file_path,
            created_at: r.created_at,
            chunk_count: 0,
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

async fn handle_export_source(
    State(state): State<Arc<AppState>>,
    Path(source_id): Path<String>,
) -> Response {
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
            chunk_count: 0,
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

async fn handle_fetch_preview(Json(payload): Json<serde_json::Value>) -> impl IntoResponse {
    let url = match payload.get("url").and_then(|v| v.as_str()) {
        Some(u) => u.to_string(),
        None => return (StatusCode::BAD_REQUEST, Json(json!({ "error": "URL is required" }))),
    };

    let client = reqwest::Client::new();
    let response = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("Failed to fetch URL: {}", e) }))),
    };

    if !response.status().is_success() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("URL returned status: {}", response.status()) })));
    }

    let html = match response.text().await {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("Failed to read response: {}", e) }))),
    };

    let content = extract_text_from_html(&html);
    let title = extract_title_from_html(&html).unwrap_or_else(|| url.clone());

    if content.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "No text content found in page" })));
    }

    (StatusCode::OK, Json(json!({
        "title": title,
        "content": content,
        "url": url
    })))
}

async fn handle_fetch_url(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<FetchUrlRequest>,
) -> impl IntoResponse {
    let client = reqwest::Client::new();
    let response = match client.get(&payload.url).send().await {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("Failed to fetch URL: {}", e) }))),
    };

    if !response.status().is_success() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("URL returned status: {}", response.status()) })));
    }

    let html = match response.text().await {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("Failed to read response: {}", e) }))),
    };

    let content = extract_text_from_html(&html);
    let title = extract_title_from_html(&html).unwrap_or_else(|| payload.url.clone());

    if content.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "No text content found in page" })));
    }

    let source_id = payload.source_id.unwrap_or_else(|| "web".to_string());
    let data_dir = std::path::Path::new(&state.data_dir);
    let mut db = state.db.write().await;
    let pipeline = IngestPipeline::new(Arc::clone(&state.embedder), Arc::clone(&state.bm25_index));

    let docs = vec![eywa::DocumentInput {
        content,
        title: Some(title.clone()),
        file_path: Some(payload.url.clone()),
        is_pdf: false,
    }];

    match pipeline.ingest_documents(&mut db, data_dir, &source_id, docs).await {
        Ok(result) => (StatusCode::OK, Json(json!({
            "title": title,
            "url": payload.url,
            "documents_created": result.documents_created,
            "chunks_created": result.chunks_created
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings & Models API
// ─────────────────────────────────────────────────────────────────────────────





/// Response for GET /api/settings
#[derive(Serialize)]
struct SettingsResponse {
    embedding_model: EmbeddingModelConfig,
    reranker_model: RerankerModelConfig,
    llm: LLMConfig,
    device: String,
    available_devices: Vec<String>,
}

/// Request for PATCH /api/settings
#[derive(Deserialize)]
struct UpdateSettingsRequest {
    #[serde(default)]
    embedding_model: Option<EmbeddingModelConfig>,
    #[serde(default)]
    reranker_model: Option<RerankerModelConfig>,
    #[serde(default)]
    llm: Option<LLMConfig>,
    #[serde(default)]
    device: Option<String>,
}

/// GET /api/settings - Get current configuration
async fn handle_get_settings() -> impl IntoResponse {
    // Get available devices based on compiled features
    let gpu_info = gpu_support_info();
    let mut available_devices = vec!["Auto".to_string(), "Cpu".to_string()];
    if gpu_info.metal_compiled {
        available_devices.push("Metal".to_string());
    }
    if gpu_info.cuda_compiled {
        available_devices.push("Cuda".to_string());
    }

    match Config::load() {
        Ok(Some(config)) => {
            let response = SettingsResponse {
                embedding_model: config.embedding_model,
                reranker_model: config.reranker_model,
                llm: config.llm,
                device: capitalize_device(config.device.name()),
                available_devices,
            };
            (StatusCode::OK, Json(json!(response)))
        }
        Ok(None) => {
            // Return defaults if no config exists
            let config = Config::default();
            let response = SettingsResponse {
                embedding_model: config.embedding_model,
                reranker_model: config.reranker_model,
                llm: config.llm,
                device: capitalize_device(config.device.name()),
                available_devices,
            };
            (StatusCode::OK, Json(json!(response)))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// PATCH /api/settings - Update configuration
async fn handle_update_settings(
    Json(payload): Json<UpdateSettingsRequest>,
) -> impl IntoResponse {
    // Load existing config or create default
    let mut config = match Config::load() {
        Ok(Some(c)) => c,
        Ok(None) => Config::default(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to load config: {}", e) })),
            )
        }
    };

    let mut model_changed = false;

    // Update embedding model if provided
    if let Some(new_model) = payload.embedding_model {
        if new_model.id != config.embedding_model.id {
            model_changed = true;
        }
        config.embedding_model = new_model;
    }

    // Update reranker model if provided
    if let Some(new_model) = payload.reranker_model {
        config.reranker_model = new_model;
    }

    // Update LLM config if provided
    if let Some(new_llm) = payload.llm {
        config.llm = new_llm;
    }

    // Update device preference if provided
    if let Some(device_str) = payload.device {
        config.device = match device_str.to_lowercase().as_str() {
            "cpu" => DevicePreference::Cpu,
            "metal" => DevicePreference::Metal,
            "cuda" => DevicePreference::Cuda,
            _ => DevicePreference::Auto,
        };
    }

    // Save config
    if let Err(e) = config.save() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to save config: {}", e) })),
        );
    }

    (
        StatusCode::OK,
        Json(json!({
            "success": true,
            "model_changed": model_changed,
            "message": if model_changed {
                "Settings saved. Embedding model changed - existing documents may need re-indexing."
            } else {
                "Settings saved successfully."
            }
        })),
    )
}

/// POST /api/chat - Generate chat completion
async fn handle_chat(
    State(_state): State<Arc<AppState>>,
    Json(messages): Json<Vec<Message>>,
) -> impl IntoResponse {
    // 1. Load config
    let config = match Config::load() {
        Ok(Some(c)) => c,
        _ => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Failed to load configuration" })),
        ),
    };

    // 2. Initialize provider
    let provider = match create_provider(&config.llm).await {
        Ok(p) => p,
        Err(e) => return (
             StatusCode::BAD_REQUEST,
             Json(json!({ "error": e.to_string() })),
        ),
    };

    // 3. Generate completion
    match provider.completion(&messages).await {
        Ok(response) => (
            StatusCode::OK,
            Json(serde_json::to_value(response).unwrap()),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// GET /api/models/embedders - List available embedding models
async fn handle_list_embedders() -> impl IntoResponse {
    let curated = EmbeddingModelConfig::curated_models();
    let downloader = ModelDownloader::new();

    // Get current selection
    let current_id = Config::load()
        .ok()
        .flatten()
        .map(|c| c.embedding_model.id);

    let models: Vec<_> = curated
        .into_iter()
        .map(|m| {
            let is_selected = current_id.as_ref() == Some(&m.id);
            let is_downloaded = downloader.is_cached(&m);
            json!({
                "id": m.id,
                "name": m.name,
                "repo_id": m.repo_id,
                "dimensions": m.dimensions,
                "size_mb": m.size_mb,
                "curated": m.curated,
                "selected": is_selected,
                "downloaded": is_downloaded
            })
        })
        .collect();

    Json(json!({
        "models": models,
        "current": current_id
    }))
}

/// GET /api/models/rerankers - List available reranker models
async fn handle_list_rerankers() -> impl IntoResponse {
    let curated = RerankerModelConfig::curated_models();
    let downloader = ModelDownloader::new();

    // Get current selection
    let current_id = Config::load()
        .ok()
        .flatten()
        .map(|c| c.reranker_model.id);

    let models: Vec<_> = curated
        .into_iter()
        .map(|m| {
            let is_selected = current_id.as_ref() == Some(&m.id);
            let is_downloaded = downloader.is_cached(&m);
            json!({
                "id": m.id,
                "name": m.name,
                "repo_id": m.repo_id,
                "size_mb": m.size_mb,
                "curated": m.curated,
                "selected": is_selected,
                "downloaded": is_downloaded
            })
        })
        .collect();

    Json(json!({
        "models": models,
        "current": current_id
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Model Download API
// ─────────────────────────────────────────────────────────────────────────────

/// Request for POST /api/models/download
#[derive(Deserialize)]
struct StartDownloadRequest {
    #[serde(rename = "type")]
    model_type: String,  // "embedder" or "reranker"
    model_id: String,
}

/// POST /api/models/download - Start downloading a model
async fn handle_start_download(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<StartDownloadRequest>,
) -> impl IntoResponse {
    // Find the model config
    let (model_name, _repo_id, size_mb) = match payload.model_type.as_str() {
        "embedder" => {
            match EmbeddingModelConfig::find_curated(&payload.model_id) {
                Some(m) => (m.name.clone(), m.repo_id.clone(), m.size_mb),
                None => return (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": format!("Embedder model '{}' not found", payload.model_id) })),
                ),
            }
        }
        "reranker" => {
            match RerankerModelConfig::find_curated(&payload.model_id) {
                Some(m) => (m.name.clone(), m.repo_id.clone(), m.size_mb),
                None => return (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": format!("Reranker model '{}' not found", payload.model_id) })),
                ),
            }
        }
        "llm" => {
            if payload.model_id == "phi3" || payload.model_id == "phi-3" {
                 let m = eywa::setup::Phi3Model::new();
                 (m.name().to_string(), m.hf_id().to_string(), m.size_mb())
            } else {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": format!("LLM model '{}' not found", payload.model_id) })),
                )
            }
        }
        _ => return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid model type. Must be 'embedder', 'reranker', or 'llm'" })),
        ),
    };

    // Check if model is already cached
    let downloader = ModelDownloader::new();
    let is_cached = match payload.model_type.as_str() {
        "embedder" => {
            let model = EmbeddingModelConfig::find_curated(&payload.model_id).unwrap();
            downloader.is_cached(&model)
        }
        "reranker" => {
            let model = RerankerModelConfig::find_curated(&payload.model_id).unwrap();
            downloader.is_cached(&model)
        }
        "llm" => {
             // For now assuming Phi-3
             let model = eywa::setup::Phi3Model::new();
             downloader.is_cached(&model)
        }
        _ => false,
    };

    if is_cached {
        return (
            StatusCode::OK,
            Json(json!({
                "status": "already_cached",
                "message": format!("Model '{}' is already downloaded", model_name)
            })),
        );
    }

    // Create job ID
    let job_id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

    // Create initial job with file placeholders
    let job = DownloadJob {
        id: job_id.clone(),
        model_type: payload.model_type.clone(),
        model_id: payload.model_id.clone(),
        model_name: model_name.clone(),
        status: DownloadStatus::Pending,
        files: vec![
            FileProgress { name: "config.json".to_string(), bytes_downloaded: 0, total_bytes: None, done: false },
            FileProgress { name: "tokenizer.json".to_string(), bytes_downloaded: 0, total_bytes: None, done: false },
            FileProgress { name: "model.safetensors".to_string(), bytes_downloaded: 0, total_bytes: None, done: false },
        ],
        started_at: now,
        completed_at: None,
        error: None,
    };

    // Store job
    {
        let mut downloads = state.downloads.lock().unwrap();
        downloads.insert(job_id.clone(), job);
    }

    // Spawn background download task
    let downloads = Arc::clone(&state.downloads);
    let model_type = payload.model_type.clone();
    let model_id = payload.model_id.clone();
    let job_id_clone = job_id.clone();

    tokio::spawn(async move {
        run_download_task(downloads, job_id_clone, model_type, model_id).await;
    });

    (
        StatusCode::ACCEPTED,
        Json(json!({
            "job_id": job_id,
            "status": "pending",
            "model_name": model_name,
            "size_mb": size_mb
        })),
    )
}

/// Background task to download a model
async fn run_download_task(
    downloads: DownloadTracker,
    job_id: String,
    model_type: String,
    model_id: String,
) {
    // Update status to downloading
    {
        let mut tracker = downloads.lock().unwrap();
        if let Some(job) = tracker.get_mut(&job_id) {
            job.status = DownloadStatus::Downloading;
        }
    }

    let downloader = ModelDownloader::new();

    // Create model tasks based on type
    let result = match model_type.as_str() {
        "embedder" => {
            let model = EmbeddingModelConfig::find_curated(&model_id).unwrap();
            download_model_files(&downloader, &downloads, &job_id, &model).await
        }
        "reranker" => {
            let model = RerankerModelConfig::find_curated(&model_id).unwrap();
            download_model_files(&downloader, &downloads, &job_id, &model).await
        }
        _ => Err(anyhow::anyhow!("Invalid model type")),
    };

    // Update final status
    let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let mut tracker = downloads.lock().unwrap();
    if let Some(job) = tracker.get_mut(&job_id) {
        job.completed_at = Some(now);
        match result {
            Ok(_) => {
                job.status = DownloadStatus::Done;
                // Mark all files as done
                for file in &mut job.files {
                    file.done = true;
                }
            }
            Err(e) => {
                job.status = DownloadStatus::Failed;
                job.error = Some(e.to_string());
            }
        }
    }
}

/// Download model files with progress tracking
async fn download_model_files<M: ModelInfo + Clone>(
    downloader: &ModelDownloader,
    downloads: &DownloadTracker,
    job_id: &str,
    model: &M,
) -> anyhow::Result<()> {

    // Create download tasks
    let mut tasks = downloader.create_tasks(model).await?;
    let model_dir = downloader.model_cache_dir(model.hf_id());
    let commit_hash = tasks.commit_hash.clone();

    // Download each file
    for file_task in &mut tasks.files {
        let file_name = file_task.name.clone();
        let job_id_owned = job_id.to_string();
        let downloads_clone = Arc::clone(downloads);

        // Progress callback
        let on_progress = move |progress: DownloadProgress| {
            let mut tracker = downloads_clone.lock().unwrap();
            if let Some(job) = tracker.get_mut(&job_id_owned) {
                // Find and update the matching file
                if let Some(file) = job.files.iter_mut().find(|f| f.name == progress.file_name) {
                    file.bytes_downloaded = progress.bytes_downloaded;
                    file.total_bytes = progress.total_bytes;
                    file.done = progress.done;
                }
            }
        };

        downloader
            .download_file(file_task, &model_dir, commit_hash.as_deref(), on_progress)
            .await?;

        // Update file as done in tracker
        {
            let mut tracker = downloads.lock().unwrap();
            if let Some(job) = tracker.get_mut(job_id) {
                if let Some(file) = job.files.iter_mut().find(|f| f.name == file_name) {
                    file.done = true;
                }
            }
        }
    }

    Ok(())
}

/// GET /api/models/download/:job_id - Get download progress
async fn handle_get_download(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let tracker = state.downloads.lock().unwrap();

    match tracker.get(&job_id) {
        Some(job) => {
            let progress = job.total_progress();
            (
                StatusCode::OK,
                Json(json!({
                    "job_id": job.id,
                    "model_type": job.model_type,
                    "model_id": job.model_id,
                    "model_name": job.model_name,
                    "status": job.status,
                    "files": job.files,
                    "total_progress": progress,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "error": job.error
                })),
            )
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Download job not found" })),
        ),
    }
}

/// GET /api/models/downloads - List all download jobs
async fn handle_list_downloads(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let tracker = state.downloads.lock().unwrap();

    let jobs: Vec<_> = tracker
        .values()
        .map(|job| {
            let progress = job.total_progress();
            json!({
                "job_id": job.id,
                "model_type": job.model_type,
                "model_id": job.model_id,
                "model_name": job.model_name,
                "status": job.status,
                "total_progress": progress,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error": job.error
            })
        })
        .collect();

    Json(json!({ "downloads": jobs }))
}

/// DELETE /api/models/cache/:model_type/:model_id - Delete a cached model
async fn handle_delete_model_cache(
    Path((model_type, model_id)): Path<(String, String)>,
) -> impl IntoResponse {
    // Get current config to check if model is selected
    let config = match Config::load() {
        Ok(Some(c)) => c,
        Ok(None) => Config::default(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to load config: {}", e) })),
            )
        }
    };

    // Check if model is currently selected
    let is_selected = match model_type.as_str() {
        "embedder" => config.embedding_model.id == model_id,
        "reranker" => config.reranker_model.id == model_id,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "Invalid model type. Must be 'embedder' or 'reranker'" })),
            )
        }
    };

    if is_selected {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Cannot delete the currently selected model" })),
        );
    }

    // Find the model config
    let (model_name, result) = match model_type.as_str() {
        "embedder" => {
            match EmbeddingModelConfig::find_curated(&model_id) {
                Some(m) => {
                    let name = m.name.clone();
                    let downloader = ModelDownloader::new();
                    (name, downloader.delete_cached(&m))
                }
                None => return (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": format!("Embedder model '{}' not found", model_id) })),
                ),
            }
        }
        "reranker" => {
            match RerankerModelConfig::find_curated(&model_id) {
                Some(m) => {
                    let name = m.name.clone();
                    let downloader = ModelDownloader::new();
                    (name, downloader.delete_cached(&m))
                }
                None => return (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": format!("Reranker model '{}' not found", model_id) })),
                ),
            }
        }
        _ => unreachable!(),
    };

    match result {
        Ok(_) => (
            StatusCode::OK,
            Json(json!({
                "success": true,
                "message": format!("Deleted cached model '{}'", model_name)
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to delete model: {}", e) })),
        ),
    }
}

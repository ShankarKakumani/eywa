//! HTTP API route handlers

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use eywa::{db, Config, ContentStore, FetchUrlRequest, IngestPipeline, IngestRequest, SearchRequest, SearchResult};
use crate::server::AppState;
use crate::utils::{create_zip, dir_size, extract_text_from_html, extract_title_from_html, lance_db_size, scan_hf_cache};

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
            "name": cfg.embedding_model.name(),
            "size_mb": cfg.embedding_model.size_mb(),
            "dimensions": cfg.embedding_model.dimensions()
        });
        response["reranker_model"] = json!({
            "name": cfg.reranker_model.name(),
            "size_mb": cfg.reranker_model.size_mb()
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

async fn handle_ingest_async(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IngestRequest>,
) -> impl IntoResponse {
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

//! MCP tool definitions and handlers

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;

use eywa::{db, ContentStore, Embedder, SearchEngine, SearchResult, VectorDB};

/// Get tool definitions for MCP tools/list response
pub fn get_tool_definitions() -> Value {
    json!([
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
    ])
}

/// Handle a tool call and return the response
/// Returns None if the response was already written to stdout (for continue cases)
pub async fn handle_tool_call(
    tool_name: &str,
    arguments: &Value,
    embedder: &Embedder,
    db: &VectorDB,
    content_store: &ContentStore,
    search_engine: &SearchEngine,
    stdout: &mut std::io::Stdout,
    id: &Option<Value>,
) -> Option<Value> {
    match tool_name {
        "search" => handle_search(arguments, embedder, db, content_store, search_engine, stdout, id).await,
        "list_sources" => handle_list_sources(db, id).await,
        "list_documents" => handle_list_documents(arguments, db, id).await,
        "get_document" => handle_get_document(arguments, db, content_store, stdout, id).await,
        "similar_docs" => handle_similar_docs(arguments, embedder, db, content_store, search_engine, stdout, id).await,
        _ => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32601, "message": format!("Unknown tool: {}", tool_name) }
        })),
    }
}

async fn handle_search(
    arguments: &Value,
    embedder: &Embedder,
    db: &VectorDB,
    content_store: &ContentStore,
    search_engine: &SearchEngine,
    stdout: &mut std::io::Stdout,
    id: &Option<Value>,
) -> Option<Value> {
    let query = arguments.get("query").and_then(|q| q.as_str()).unwrap_or("");
    let limit = arguments.get("limit").and_then(|l| l.as_u64()).unwrap_or(5) as usize;
    let source = arguments.get("source").and_then(|s| s.as_str());

    match embedder.embed(query) {
        Ok(embedding) => {
            match db.search_filtered(&embedding, limit * 2, source).await {
                Ok(chunk_metas) => {
                    let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
                    let contents = match content_store.get_chunks(&chunk_ids) {
                        Ok(c) => c,
                        Err(e) => {
                            let resp = json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": { "code": -32000, "message": format!("Content fetch error: {}", e) }
                            });
                            writeln!(stdout, "{}", resp).ok();
                            stdout.flush().ok();
                            return None;
                        }
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

                    Some(json!({
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
                    }))
                }
                Err(e) => Some(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32000, "message": format!("Search error: {}", e) }
                }))
            }
        }
        Err(e) => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32000, "message": format!("Embedding error: {}", e) }
        }))
    }
}

async fn handle_list_sources(db: &VectorDB, id: &Option<Value>) -> Option<Value> {
    match db.list_sources().await {
        Ok(sources) => {
            let text = if sources.is_empty() {
                "No sources found in the knowledge base.".to_string()
            } else {
                sources.iter().map(|s| {
                    format!("- {} ({} chunks)", s.name, s.chunk_count)
                }).collect::<Vec<_>>().join("\n")
            };

            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": format!("Sources:\n{}", text)
                    }]
                }
            }))
        }
        Err(e) => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32000, "message": format!("Error: {}", e) }
        }))
    }
}

async fn handle_list_documents(arguments: &Value, db: &VectorDB, id: &Option<Value>) -> Option<Value> {
    let source_id = arguments.get("source_id").and_then(|s| s.as_str()).unwrap_or("");

    if source_id.is_empty() {
        return Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32602, "message": "source_id is required" }
        }));
    }

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

            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": format!("Documents in '{}':\n{}", source_id, text)
                    }]
                }
            }))
        }
        Err(e) => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32000, "message": format!("Error: {}", e) }
        }))
    }
}

async fn handle_get_document(
    arguments: &Value,
    db: &VectorDB,
    content_store: &ContentStore,
    stdout: &mut std::io::Stdout,
    id: &Option<Value>,
) -> Option<Value> {
    let doc_id = arguments.get("document_id").and_then(|s| s.as_str()).unwrap_or("");

    if doc_id.is_empty() {
        return Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32602, "message": "document_id is required" }
        }));
    }

    match db.get_document(doc_id).await {
        Ok(Some(record)) => {
            let content = match content_store.get_document(doc_id) {
                Ok(Some(c)) => c,
                Ok(None) => {
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32000, "message": format!("Document content not found: {}", doc_id) }
                    });
                    writeln!(stdout, "{}", resp).ok();
                    stdout.flush().ok();
                    return None;
                }
                Err(e) => {
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32000, "message": format!("Content fetch error: {}", e) }
                    });
                    writeln!(stdout, "{}", resp).ok();
                    stdout.flush().ok();
                    return None;
                }
            };

            let file_info = record.file_path.as_ref()
                .map(|p| format!("\nFile: {}", p))
                .unwrap_or_default();

            Some(json!({
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
            }))
        }
        Ok(None) => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32000, "message": format!("Document not found: {}", doc_id) }
        })),
        Err(e) => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32000, "message": format!("Error: {}", e) }
        }))
    }
}

async fn handle_similar_docs(
    arguments: &Value,
    embedder: &Embedder,
    db: &VectorDB,
    content_store: &ContentStore,
    search_engine: &SearchEngine,
    stdout: &mut std::io::Stdout,
    id: &Option<Value>,
) -> Option<Value> {
    let doc_id = arguments.get("document_id").and_then(|s| s.as_str()).unwrap_or("");
    let limit = arguments.get("limit").and_then(|l| l.as_u64()).unwrap_or(5) as usize;

    if doc_id.is_empty() {
        return Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32602, "message": "document_id is required" }
        }));
    }

    // Get source document content
    let source_content = match content_store.get_document(doc_id) {
        Ok(Some(c)) => c,
        Ok(None) => {
            let resp = json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32000, "message": format!("Document not found: {}", doc_id) }
            });
            writeln!(stdout, "{}", resp).ok();
            stdout.flush().ok();
            return None;
        }
        Err(e) => {
            let resp = json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32000, "message": format!("Error fetching document: {}", e) }
            });
            writeln!(stdout, "{}", resp).ok();
            stdout.flush().ok();
            return None;
        }
    };

    // Embed the source document
    match embedder.embed(&source_content) {
        Ok(embedding) => {
            match db.search(&embedding, (limit + 5) * 2).await {
                Ok(chunk_metas) => {
                    // Filter out chunks from the same document
                    let chunk_metas: Vec<_> = chunk_metas
                        .into_iter()
                        .filter(|c| c.document_id != doc_id)
                        .collect();

                    let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
                    let contents = match content_store.get_chunks(&chunk_ids) {
                        Ok(c) => c,
                        Err(e) => {
                            let resp = json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": { "code": -32000, "message": format!("Content fetch error: {}", e) }
                            });
                            writeln!(stdout, "{}", resp).ok();
                            stdout.flush().ok();
                            return None;
                        }
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

                    Some(json!({
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
                    }))
                }
                Err(e) => Some(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32000, "message": format!("Search error: {}", e) }
                }))
            }
        }
        Err(e) => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32000, "message": format!("Embedding error: {}", e) }
        }))
    }
}

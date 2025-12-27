//! MCP (Model Context Protocol) server module
//! Provides JSON-RPC interface for Claude/Cursor integration

mod tools;

use anyhow::Result;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};

use eywa::{ContentStore, Embedder, SearchEngine, VectorDB};
use tools::{get_tool_definitions, handle_tool_call};

/// Run the MCP server (JSON-RPC over stdio)
pub async fn run_mcp_server(data_dir: &str) -> Result<()> {
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
                        "tools": get_tool_definitions()
                    }
                })
            }

            "tools/call" => {
                let params = request.get("params").cloned().unwrap_or(json!({}));
                let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

                match handle_tool_call(
                    tool_name,
                    &arguments,
                    &embedder,
                    &db,
                    &content_store,
                    &search_engine,
                    &mut stdout,
                    &id,
                ).await {
                    Some(resp) => resp,
                    None => continue, // Response already written by handler
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

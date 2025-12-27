//! Search command handler

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use eywa::{ContentStore, Embedder, SearchEngine, SearchResult, VectorDB};

pub async fn run_search(data_dir: &str, query: &str, limit: usize) -> Result<()> {
    let embedder = Embedder::new()?;
    let db = VectorDB::new(data_dir).await?;
    let content_store = ContentStore::open(&Path::new(data_dir).join("content.db"))?;
    let search_engine = SearchEngine::with_reranker()?;

    println!("Searching for: {}\n", query);

    let query_embedding = embedder.embed(query)?;
    let chunk_metas = db.search(&query_embedding, 50).await?;

    // Fetch content from SQLite
    let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
    let contents = content_store.get_chunks(&chunk_ids)?;
    let content_map: HashMap<String, String> = contents.into_iter().collect();

    // Combine metadata + content
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

    if results.is_empty() {
        println!("No results found.");
    } else {
        for (i, result) in results.iter().take(limit).enumerate() {
            println!("{}. [Score: {:.3}]", i + 1, result.score);
            if let Some(ref title) = result.title {
                println!("   Title: {}", title);
            }
            if let Some(ref file_path) = result.file_path {
                print!("   File: {}", file_path);
                if let Some(line) = result.line_start {
                    print!(":{}", line);
                }
                println!();
            }
            println!("   Source: {}", result.source_id);

            // Show first 200 chars of content
            let preview: String = result
                .content
                .chars()
                .take(200)
                .collect();
            println!("   Preview: {}...\n", preview.replace('\n', " "));
        }
    }

    Ok(())
}

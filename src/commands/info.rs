//! Info and storage command handlers

use anyhow::Result;
use eywa::{gpu_support_info, Config};
use crate::utils::{dir_size, format_bytes, lance_db_size, scan_hf_cache};
use std::path::Path;

pub fn run_info(data_dir: &str) -> Result<()> {
    println!("Eywa - Personal Knowledge Base\n");

    match Config::load()? {
        Some(config) => {
            println!("Embedding: {} ({}MB, {} dims)",
                config.embedding_model.name(),
                config.embedding_model.size_mb(),
                config.embedding_model.dimensions()
            );
            println!("Reranker:  {} ({}MB)",
                config.reranker_model.name(),
                config.reranker_model.size_mb()
            );
            println!("Device:    {} (preference: {})",
                config.device.name(),
                config.device.name()
            );
        }
        None => {
            println!("Not initialized. Run 'eywa' or 'eywa init' to set up.");
        }
    }

    // Show GPU support info
    let gpu_info = gpu_support_info();
    println!("\nGPU Support: {}", gpu_info.summary());
    if !gpu_info.any_gpu() {
        println!("  Rebuild with --features metal (macOS) or --features cuda (NVIDIA)");
    }

    println!("\nDatabase: LanceDB (file-based)");
    println!("Data directory: {}", data_dir);

    Ok(())
}

pub fn run_storage(data_dir: &str) -> Result<()> {
    println!("Eywa Storage Usage\n");

    // Data storage
    let data_path = Path::new(data_dir);
    let content_db_bytes = std::fs::metadata(data_path.join("content.db"))
        .map(|m| m.len())
        .unwrap_or(0);
    let vector_db_bytes = lance_db_size(data_path);
    let bm25_index_bytes = dir_size(&data_path.join("tantivy")).unwrap_or(0);
    let data_total = content_db_bytes + vector_db_bytes + bm25_index_bytes;

    println!("\x1b[1mData\x1b[0m");
    println!("  Content DB (SQLite)    {:>12}", format_bytes(content_db_bytes));
    println!("  Vector DB (LanceDB)    {:>12}", format_bytes(vector_db_bytes));
    println!("  BM25 Index (Tantivy)   {:>12}", format_bytes(bm25_index_bytes));
    println!("  \x1b[90m───────────────────────────────\x1b[0m");
    println!("  Subtotal               {:>12}", format_bytes(data_total));

    // Models storage (scan HuggingFace cache)
    let cached_models = scan_hf_cache();
    let models_total: u64 = cached_models.iter().map(|m| m.size_bytes).sum();

    println!("\n\x1b[1mModels\x1b[0m (cached from HuggingFace)");
    if cached_models.is_empty() {
        println!("  No models downloaded yet");
    } else {
        for model in &cached_models {
            println!("  {:<24} {:>12}", model.name, format_bytes(model.size_bytes));
        }
        println!("  \x1b[90m───────────────────────────────\x1b[0m");
        println!("  Subtotal               {:>12}", format_bytes(models_total));
    }

    // Total
    let grand_total = data_total + models_total;
    println!("\n\x1b[1m═══════════════════════════════════\x1b[0m");
    println!("\x1b[1mTotal                    {:>12}\x1b[0m", format_bytes(grand_total));

    Ok(())
}

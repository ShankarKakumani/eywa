//! Utility functions for Eywa CLI

use anyhow::Result;
use std::path::Path;

/// Expand ~ to home directory in paths
pub fn expand_path(path: &str) -> String {
    if path.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return path.replacen("~", &home, 1);
        }
    }
    path.to_string()
}

/// Format bytes into human-readable string
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Create a zip file from documents
pub fn create_zip(docs: &[eywa::Document]) -> Result<Vec<u8>> {
    use std::io::{Cursor, Write};
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    let mut buffer = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(&mut buffer);
    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    for doc in docs {
        // Create path: source_id/title (sanitize for filesystem)
        let safe_title = doc.title
            .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        let path = format!("{}/{}", doc.source_id, safe_title);

        zip.start_file(&path, options)?;
        zip.write_all(doc.content.as_bytes())?;
    }

    zip.finish()?;
    Ok(buffer.into_inner())
}

/// Extract text content from HTML and convert to Markdown
pub fn extract_text_from_html(html: &str) -> String {
    html2md::rewrite_html(html, false)
}

/// Extract title from HTML
pub fn extract_title_from_html(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<title>")?;
    let end = lower[start..].find("</title>")?;
    let title = &html[start + 7..start + end];
    let title = title.trim();
    if title.is_empty() {
        None
    } else {
        Some(title.to_string())
    }
}

/// Calculate total size of a directory recursively
pub fn dir_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                total += dir_size(&path)?;
            } else {
                total += entry.metadata()?.len();
            }
        }
    }
    Ok(total)
}

/// Calculate total size of all LanceDB table directories (.lance) in a path
pub fn lance_db_size(data_path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(data_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.extension().map_or(false, |ext| ext == "lance") {
                total += dir_size(&path).unwrap_or(0);
            }
        }
    }
    total
}

/// Model info from HuggingFace cache
#[derive(Debug, Clone, serde::Serialize)]
pub struct CachedModel {
    pub name: String,
    pub size_bytes: u64,
}

/// Scan HuggingFace cache directory and return all downloaded models
pub fn scan_hf_cache() -> Vec<CachedModel> {
    let mut models = Vec::new();

    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return models,
    };

    let hf_cache = home.join(".cache").join("huggingface").join("hub");
    if !hf_cache.exists() {
        return models;
    }

    // HuggingFace cache structure: ~/.cache/huggingface/hub/models--org--name/
    if let Ok(entries) = std::fs::read_dir(&hf_cache) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let dir_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                // Only process model directories (models--org--name)
                if dir_name.starts_with("models--") {
                    // Parse model name from directory: models--org--name -> org/name
                    let name = dir_name
                        .strip_prefix("models--")
                        .unwrap_or(dir_name)
                        .replace("--", "/");

                    let size = dir_size(&path).unwrap_or(0);

                    models.push(CachedModel {
                        name,
                        size_bytes: size,
                    });
                }
            }
        }
    }

    // Sort by size (largest first)
    models.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    models
}

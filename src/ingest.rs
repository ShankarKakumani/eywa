//! Document ingestion with chunking and hybrid storage
//!
//! Ingests documents by:
//! 1. Storing content in SQLite (compressed)
//! 2. Storing metadata + vectors in LanceDB

use anyhow::Result;
use std::path::Path;
use walkdir::WalkDir;

use crate::content::ContentStore;
use crate::db::{ChunkRecord, VectorDB};
use crate::embed::Embedder;
use crate::types::{DocumentInput, DocumentRecord, IngestResponse};

const CHUNK_SIZE: usize = 1000;
const CHUNK_OVERLAP: usize = 200;
const BATCH_SIZE: usize = 32;

/// Intermediate chunk structure during ingestion
struct ChunkData {
    id: String,
    document_id: String,
    source_id: String,
    title: Option<String>,
    content: String,
    file_path: Option<String>,
    line_start: u32,
    line_end: u32,
    content_hash: String,
    // Hierarchical metadata (defaults for legacy chunker)
    section: Option<String>,
    subsection: Option<String>,
    hierarchy: Vec<String>,
    has_code: bool,
}

pub struct Ingester<'a> {
    embedder: &'a Embedder,
}

impl<'a> Ingester<'a> {
    pub fn new(embedder: &'a Embedder) -> Self {
        Self { embedder }
    }

    /// Check if file extension is supported
    fn is_supported_extension(ext: &str) -> bool {
        matches!(
            ext,
            "md" | "txt"
                | "rs"
                | "py"
                | "js"
                | "ts"
                | "tsx"
                | "jsx"
                | "go"
                | "java"
                | "c"
                | "cpp"
                | "h"
                | "hpp"
                | "json"
                | "yaml"
                | "yml"
                | "toml"
                | "xml"
                | "html"
                | "css"
                | "scss"
                | "sql"
                | "sh"
                | "bash"
                | "zsh"
                | "fish"
                | "dart"
                | "swift"
                | "kt"
                | "kts"
                | "rb"
                | "php"
                | "vue"
                | "svelte"
        )
    }

    /// Chunk text into smaller pieces with overlap
    fn chunk_text(
        content: &str,
        document_id: &str,
        source_id: &str,
        title: Option<&str>,
        file_path: Option<&str>,
    ) -> Vec<ChunkData> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        if lines.is_empty() {
            return chunks;
        }

        let mut current_chunk = String::new();
        let mut chunk_start_line = 1u32;
        let mut current_line = 1u32;

        for line in lines {
            let line_with_newline = format!("{}\n", line);

            if current_chunk.len() + line_with_newline.len() > CHUNK_SIZE && !current_chunk.is_empty()
            {
                chunks.push(Self::create_chunk(
                    &current_chunk,
                    document_id,
                    source_id,
                    title,
                    file_path,
                    chunk_start_line,
                    current_line - 1,
                ));

                // Overlap: keep last CHUNK_OVERLAP chars
                let target_start = current_chunk.len().saturating_sub(CHUNK_OVERLAP);
                let overlap_start = current_chunk
                    .char_indices()
                    .map(|(i, _)| i)
                    .find(|&i| i >= target_start)
                    .unwrap_or(0);
                current_chunk = current_chunk[overlap_start..].to_string();
                chunk_start_line =
                    current_line.saturating_sub(current_chunk.lines().count() as u32);
            }

            current_chunk.push_str(&line_with_newline);
            current_line += 1;
        }

        // Last chunk
        if !current_chunk.trim().is_empty() {
            chunks.push(Self::create_chunk(
                &current_chunk,
                document_id,
                source_id,
                title,
                file_path,
                chunk_start_line,
                current_line - 1,
            ));
        }

        chunks
    }

    fn create_chunk(
        content: &str,
        document_id: &str,
        source_id: &str,
        title: Option<&str>,
        file_path: Option<&str>,
        line_start: u32,
        line_end: u32,
    ) -> ChunkData {
        // Detect if content contains code blocks
        let has_code = content.contains("```");

        ChunkData {
            id: uuid::Uuid::new_v4().to_string(),
            document_id: document_id.to_string(),
            source_id: source_id.to_string(),
            title: title.map(|s| s.to_string()),
            content: content.to_string(),
            file_path: file_path.map(|s| s.to_string()),
            line_start,
            line_end,
            content_hash: format!("{:x}", md5::compute(content.as_bytes())),
            // Legacy chunker: no hierarchy, just detect code blocks
            section: None,
            subsection: None,
            hierarchy: Vec::new(),
            has_code,
        }
    }

    fn now_iso() -> String {
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
    }

    /// Ingest documents with hybrid storage
    ///
    /// Takes data_dir path and opens SQLite connection internally to avoid
    /// holding non-Send types across await points.
    pub async fn ingest_documents(
        &self,
        db: &mut VectorDB,
        data_dir: &Path,
        source_id: &str,
        documents: Vec<DocumentInput>,
    ) -> Result<IngestResponse> {
        let mut documents_created = 0u32;
        let mut chunks_created = 0u32;
        let mut chunks_skipped = 0u32;
        let mut document_ids = Vec::new();

        // Prepare all data upfront
        struct PreparedDoc {
            id: String,
            source_id: String,
            content: String,
            title: String,
            file_path: Option<String>,
            created_at: String,
            content_length: u32,
            chunks: Vec<ChunkData>,
        }

        let mut prepared_docs = Vec::new();
        for doc_input in documents {
            if doc_input.content.trim().is_empty() {
                continue;
            }

            let doc_id = uuid::Uuid::new_v4().to_string();
            let title = doc_input
                .title
                .clone()
                .unwrap_or_else(|| format!("Untitled-{}", &doc_id[..8]));
            let created_at = Self::now_iso();
            let content_length = doc_input.content.len() as u32;

            let chunks = Self::chunk_text(
                &doc_input.content,
                &doc_id,
                source_id,
                Some(&title),
                doc_input.file_path.as_deref(),
            );

            prepared_docs.push(PreparedDoc {
                id: doc_id,
                source_id: source_id.to_string(),
                content: doc_input.content,
                title,
                file_path: doc_input.file_path,
                created_at,
                content_length,
                chunks,
            });
        }

        // Phase 1: All SQLite operations (in a block that doesn't cross await)
        {
            let content_store = ContentStore::open(&data_dir.join("content.db"))?;
            for doc in &prepared_docs {
                content_store.insert_document(
                    &doc.id,
                    &doc.source_id,
                    &doc.title,
                    doc.file_path.as_deref(),
                    &doc.content,
                    &doc.created_at,
                )?;

                // Collect chunk contents
                let chunk_contents: Vec<(String, String, String)> = doc
                    .chunks
                    .iter()
                    .map(|c| (c.id.clone(), c.document_id.clone(), c.content.clone()))
                    .collect();

                if !chunk_contents.is_empty() {
                    content_store.insert_chunks(&chunk_contents)?;
                }
            }
            // content_store is dropped here
        }

        // Phase 2: All LanceDB operations (async)
        for doc in prepared_docs {
            let chunk_count = doc.chunks.len() as u32;

            // Store document metadata in LanceDB
            let doc_record = DocumentRecord {
                id: doc.id.clone(),
                source_id: source_id.to_string(),
                title: doc.title.clone(),
                file_path: doc.file_path.clone(),
                created_at: doc.created_at,
                chunk_count,
                content_length: doc.content_length,
            };
            db.insert_document(&doc_record).await?;

            documents_created += 1;
            document_ids.push(doc.id.clone());

            // Process chunks in batches
            for batch in doc.chunks.chunks(BATCH_SIZE) {
                let mut new_chunks = Vec::new();
                let mut texts = Vec::new();

                for chunk in batch {
                    // Skip duplicates
                    if db.chunk_exists(&chunk.content_hash).await? {
                        chunks_skipped += 1;
                        continue;
                    }

                    texts.push(chunk.content.clone());
                    new_chunks.push(chunk);
                }

                if new_chunks.is_empty() {
                    continue;
                }

                // Generate embeddings
                let embeddings = self.embedder.embed_batch(&texts)?;

                // Store chunk metadata + vectors in LanceDB
                let chunk_records: Vec<ChunkRecord> = new_chunks
                    .iter()
                    .map(|c| ChunkRecord {
                        id: c.id.clone(),
                        document_id: c.document_id.clone(),
                        source_id: c.source_id.clone(),
                        title: c.title.clone(),
                        file_path: c.file_path.clone(),
                        line_start: Some(c.line_start),
                        line_end: Some(c.line_end),
                        content_hash: c.content_hash.clone(),
                        // Hierarchical metadata from legacy chunker
                        section: c.section.clone(),
                        subsection: c.subsection.clone(),
                        hierarchy: c.hierarchy.clone(),
                        has_code: c.has_code,
                    })
                    .collect();

                db.insert_chunks(&chunk_records, &embeddings).await?;
                chunks_created += new_chunks.len() as u32;
            }
        }

        Ok(IngestResponse {
            source_id: source_id.to_string(),
            documents_created,
            chunks_created,
            chunks_skipped,
            document_ids,
        })
    }

    /// Ingest from file path (CLI)
    pub async fn ingest_from_path(
        &self,
        db: &mut VectorDB,
        data_dir: &Path,
        source_id: &str,
        file_path: &str,
    ) -> Result<IngestResponse> {
        let path = Path::new(file_path);

        let files: Vec<_> = if path.is_dir() {
            WalkDir::new(file_path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let p = e.path();
                    if !p.is_file() {
                        return false;
                    }
                    let ext = p
                        .extension()
                        .map(|e| e.to_string_lossy().to_lowercase())
                        .unwrap_or_default();
                    Self::is_supported_extension(&ext)
                })
                .map(|e| e.path().to_path_buf())
                .collect()
        } else {
            vec![path.to_path_buf()]
        };

        let mut doc_inputs = Vec::new();
        for file in &files {
            let content = match std::fs::read_to_string(file) {
                Ok(c) if !c.trim().is_empty() => c,
                _ => continue,
            };

            doc_inputs.push(DocumentInput {
                content,
                title: file.file_name().map(|n| n.to_string_lossy().to_string()),
                file_path: Some(file.to_string_lossy().to_string()),
            });
        }

        self.ingest_documents(db, data_dir, source_id, doc_inputs)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_extensions_code() {
        // Code files
        assert!(Ingester::is_supported_extension("rs"));
        assert!(Ingester::is_supported_extension("py"));
        assert!(Ingester::is_supported_extension("js"));
        assert!(Ingester::is_supported_extension("ts"));
        assert!(Ingester::is_supported_extension("tsx"));
        assert!(Ingester::is_supported_extension("go"));
        assert!(Ingester::is_supported_extension("java"));
        assert!(Ingester::is_supported_extension("c"));
        assert!(Ingester::is_supported_extension("cpp"));
    }

    #[test]
    fn test_supported_extensions_docs() {
        // Documentation files
        assert!(Ingester::is_supported_extension("md"));
        assert!(Ingester::is_supported_extension("txt"));
    }

    #[test]
    fn test_supported_extensions_config() {
        // Config files
        assert!(Ingester::is_supported_extension("json"));
        assert!(Ingester::is_supported_extension("yaml"));
        assert!(Ingester::is_supported_extension("yml"));
        assert!(Ingester::is_supported_extension("toml"));
    }

    #[test]
    fn test_unsupported_extensions() {
        // Binary/media files
        assert!(!Ingester::is_supported_extension("exe"));
        assert!(!Ingester::is_supported_extension("pdf"));
        assert!(!Ingester::is_supported_extension("png"));
        assert!(!Ingester::is_supported_extension("jpg"));
        assert!(!Ingester::is_supported_extension("mp4"));
        assert!(!Ingester::is_supported_extension("zip"));
        assert!(!Ingester::is_supported_extension(""));
    }
}

//! BM25 keyword search using Tantivy
//!
//! Provides full-text search alongside vector search for hybrid retrieval.

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Mutex;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

/// Result from BM25 search
#[derive(Debug, Clone)]
pub struct BM25Result {
    pub chunk_id: String,
    pub score: f32,
}

/// BM25 search index using Tantivy
pub struct BM25Index {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    // Schema fields
    chunk_id_field: Field,
    source_id_field: Field,
    content_field: Field,
    title_field: Field,
}

/// Remove stale Tantivy lock files that may be left after a crash
fn remove_stale_locks(index_path: &Path) {
    let _ = std::fs::remove_file(index_path.join(".tantivy-meta.lock"));
    let _ = std::fs::remove_file(index_path.join(".tantivy-writer.lock"));
}

impl BM25Index {
    /// Open or create a BM25 index at the given path
    pub fn open(data_dir: &Path) -> Result<Self> {
        let index_path = data_dir.join("tantivy");
        std::fs::create_dir_all(&index_path)
            .with_context(|| format!("Failed to create tantivy dir at {:?}", index_path))?;

        // Define schema
        let mut schema_builder = Schema::builder();
        let chunk_id_field = schema_builder.add_text_field("chunk_id", STRING | STORED);
        let source_id_field = schema_builder.add_text_field("source_id", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT);
        let title_field = schema_builder.add_text_field("title", TEXT);
        let schema = schema_builder.build();

        // Open or create index
        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path)
                .with_context(|| "Failed to open existing tantivy index")?
        } else {
            Index::create_in_dir(&index_path, schema.clone())
                .with_context(|| "Failed to create tantivy index")?
        };

        // Create reader with auto-reload
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to create index reader")?;

        // Create writer with 50MB heap - retry once if lock is stale
        let writer = match index.writer(50_000_000) {
            Ok(w) => w,
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("Lockfile") || err_str.contains("LockBusy") {
                    eprintln!("Warning: Removing stale Tantivy lock files and retrying...");
                    remove_stale_locks(&index_path);
                    index
                        .writer(50_000_000)
                        .context("Failed to create index writer after removing stale locks")?
                } else {
                    return Err(e).context("Failed to create index writer");
                }
            }
        };

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            chunk_id_field,
            source_id_field,
            content_field,
            title_field,
        })
    }

    /// Index a batch of chunks
    pub fn index_chunks(&self, chunks: &[ChunkInput]) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();

        for chunk in chunks {
            let mut doc = TantivyDocument::default();
            doc.add_text(self.chunk_id_field, &chunk.id);
            doc.add_text(self.source_id_field, &chunk.source_id);
            doc.add_text(self.content_field, &chunk.content);
            if let Some(ref title) = chunk.title {
                doc.add_text(self.title_field, title);
            }
            writer.add_document(doc)?;
        }

        writer.commit().context("Failed to commit tantivy index")?;
        drop(writer); // Release lock before reload
        // Force reader reload to see new documents immediately
        self.reader.reload().context("Failed to reload index reader")?;
        Ok(())
    }

    /// Search for chunks matching the query
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<BM25Result>> {
        let searcher = self.reader.searcher();

        // Parse query across content and title fields
        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.content_field, self.title_field],
        );

        // Handle empty or invalid queries gracefully
        let query = match query_parser.parse_query(query) {
            Ok(q) => q,
            Err(_) => {
                // If query parsing fails, return empty results
                return Ok(vec![]);
            }
        };

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .context("Tantivy search failed")?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .context("Failed to retrieve document")?;

            if let Some(chunk_id) = doc.get_first(self.chunk_id_field) {
                if let Some(chunk_id_str) = chunk_id.as_str() {
                    results.push(BM25Result {
                        chunk_id: chunk_id_str.to_string(),
                        score,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Search within a specific source
    pub fn search_source(&self, query: &str, source_id: &str, limit: usize) -> Result<Vec<BM25Result>> {
        let searcher = self.reader.searcher();

        // Build a combined query: content/title match AND source filter
        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.content_field, self.title_field],
        );

        let content_query = match query_parser.parse_query(query) {
            Ok(q) => q,
            Err(_) => return Ok(vec![]),
        };

        // Create source filter
        let source_term = tantivy::Term::from_field_text(self.source_id_field, source_id);
        let source_query = tantivy::query::TermQuery::new(
            source_term,
            tantivy::schema::IndexRecordOption::Basic,
        );

        // Combine with AND
        let combined_query = tantivy::query::BooleanQuery::new(vec![
            (tantivy::query::Occur::Must, content_query),
            (tantivy::query::Occur::Must, Box::new(source_query)),
        ]);

        let top_docs = searcher
            .search(&combined_query, &TopDocs::with_limit(limit))
            .context("Tantivy search failed")?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .context("Failed to retrieve document")?;

            if let Some(chunk_id) = doc.get_first(self.chunk_id_field) {
                if let Some(chunk_id_str) = chunk_id.as_str() {
                    results.push(BM25Result {
                        chunk_id: chunk_id_str.to_string(),
                        score,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Delete all chunks for a source
    pub fn delete_source(&self, source_id: &str) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        let source_term = tantivy::Term::from_field_text(self.source_id_field, source_id);
        writer.delete_term(source_term);
        writer.commit().context("Failed to commit deletion")?;
        drop(writer); // Release lock before reload
        self.reader.reload().context("Failed to reload index reader")?;
        Ok(())
    }

    /// Delete a specific chunk by ID
    pub fn delete_chunk(&self, chunk_id: &str) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        let chunk_term = tantivy::Term::from_field_text(self.chunk_id_field, chunk_id);
        writer.delete_term(chunk_term);
        writer.commit().context("Failed to commit deletion")?;
        drop(writer); // Release lock before reload
        self.reader.reload().context("Failed to reload index reader")?;
        Ok(())
    }

    /// Clear all documents from the index
    pub fn reset(&self) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        writer.delete_all_documents()?;
        writer.commit().context("Failed to commit reset")?;
        drop(writer); // Release lock before reload
        self.reader.reload().context("Failed to reload index reader")?;
        Ok(())
    }
}

/// Input for indexing a chunk
#[derive(Debug, Clone)]
pub struct ChunkInput {
    pub id: String,
    pub source_id: String,
    pub content: String,
    pub title: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_bm25_index_and_search() {
        let temp_dir = TempDir::new().unwrap();
        let index = BM25Index::open(temp_dir.path()).unwrap();

        // Index some chunks
        let chunks = vec![
            ChunkInput {
                id: "chunk1".to_string(),
                source_id: "docs".to_string(),
                content: "JWT authentication uses tokens for stateless auth".to_string(),
                title: Some("Auth Guide".to_string()),
            },
            ChunkInput {
                id: "chunk2".to_string(),
                source_id: "docs".to_string(),
                content: "OAuth2 is an authorization framework".to_string(),
                title: Some("OAuth Guide".to_string()),
            },
            ChunkInput {
                id: "chunk3".to_string(),
                source_id: "code".to_string(),
                content: "fn authenticate(token: &str) -> bool { true }".to_string(),
                title: Some("auth.rs".to_string()),
            },
        ];

        index.index_chunks(&chunks).unwrap();

        // Search for JWT
        let results = index.search("JWT authentication", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "chunk1");

        // Search for OAuth
        let results = index.search("OAuth authorization", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "chunk2");

        // Search within source
        let results = index.search_source("authenticate", "code", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "chunk3");

        // Search within wrong source should return empty
        let results = index.search_source("JWT", "code", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_delete() {
        let temp_dir = TempDir::new().unwrap();
        let index = BM25Index::open(temp_dir.path()).unwrap();

        let chunks = vec![
            ChunkInput {
                id: "chunk1".to_string(),
                source_id: "docs".to_string(),
                content: "Test document one".to_string(),
                title: None,
            },
            ChunkInput {
                id: "chunk2".to_string(),
                source_id: "docs".to_string(),
                content: "Test document two".to_string(),
                title: None,
            },
        ];

        index.index_chunks(&chunks).unwrap();

        // Verify both exist
        let results = index.search("test document", 10).unwrap();
        assert_eq!(results.len(), 2);

        // Delete source
        index.delete_source("docs").unwrap();

        // Reload reader to see changes
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Verify deleted
        let results = index.search("test document", 10).unwrap();
        assert!(results.is_empty());
    }
}

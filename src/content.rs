//! SQLite-based content storage with zstd compression.
//!
//! Stores document and chunk content separately from vector storage.
//! This enables efficient storage (content stored once, compressed)
//! while keeping vector search fast (LanceDB handles only embeddings).

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

/// Compression level for zstd (1-22, higher = smaller but slower)
const COMPRESSION_LEVEL: i32 = 3;

/// Document row returned from streaming iteration.
#[derive(Debug, Clone)]
pub struct DocumentRow {
    pub id: String,
    pub source_id: String,
    pub title: String,
    pub file_path: Option<String>,
    pub content: String,
    pub created_at: String,
}

/// Document metadata (without content) for listing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentListItem {
    pub id: String,
    pub source_id: String,
    pub title: String,
    pub file_path: Option<String>,
    pub content_length: usize,
    pub created_at: String,
}

/// Source stats for web UI listing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SourceStats {
    pub id: String,
    pub doc_count: u64,
    pub total_size: u64,
    pub last_updated: Option<String>,
}

/// Content store backed by SQLite with zstd compression.
pub struct ContentStore {
    conn: Connection,
}

impl ContentStore {
    /// Open or create a content store at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open content store at {:?}", path))?;

        let store = Self { conn };
        store.init_schema()?;
        store.migrate_schema()?;

        Ok(store)
    }

    /// Initialize database schema.
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                source_id   TEXT NOT NULL DEFAULT 'unknown',
                title       TEXT NOT NULL DEFAULT 'Untitled',
                file_path   TEXT,
                content     BLOB NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                content     BLOB NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id);

            CREATE INDEX IF NOT EXISTS idx_documents_source
                ON documents(source_id);

            PRAGMA foreign_keys = ON;
            ",
        )?;

        Ok(())
    }

    /// Migrate existing databases to new schema (add source_id, title, file_path columns).
    fn migrate_schema(&self) -> Result<()> {
        // Check if source_id column exists
        let has_source_id: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('documents') WHERE name='source_id'",
            [],
            |row| row.get(0),
        )?;

        if has_source_id == 0 {
            // Old schema - need to migrate
            self.conn.execute_batch(
                "
                ALTER TABLE documents ADD COLUMN source_id TEXT NOT NULL DEFAULT 'unknown';
                ALTER TABLE documents ADD COLUMN title TEXT NOT NULL DEFAULT 'Untitled';
                ALTER TABLE documents ADD COLUMN file_path TEXT;
                CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
                ",
            )?;
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Document Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Store a document's content with full metadata.
    pub fn insert_document(
        &self,
        id: &str,
        source_id: &str,
        title: &str,
        file_path: Option<&str>,
        content: &str,
        created_at: &str,
    ) -> Result<()> {
        let compressed = compress(content)?;

        self.conn.execute(
            "INSERT OR REPLACE INTO documents (id, source_id, title, file_path, content, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, source_id, title, file_path, compressed, created_at],
        )?;

        Ok(())
    }

    /// Get a document's content by ID.
    pub fn get_document(&self, id: &str) -> Result<Option<String>> {
        let result: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT content FROM documents WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?;

        match result {
            Some(compressed) => Ok(Some(decompress(&compressed)?)),
            None => Ok(None),
        }
    }

    /// Delete a document and its chunks.
    pub fn delete_document(&self, id: &str) -> Result<()> {
        // Chunks are deleted via CASCADE
        self.conn
            .execute("DELETE FROM documents WHERE id = ?1", params![id])?;

        Ok(())
    }

    /// Count total documents.
    pub fn count_documents(&self) -> Result<u64> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents",
            [],
            |row| row.get(0),
        )?;
        Ok(count as u64)
    }

    /// List documents for a source (metadata only, no content decompression).
    /// Returns (documents, total_count) for pagination.
    pub fn list_documents_by_source(
        &self,
        source_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<(Vec<DocumentListItem>, usize)> {
        // Get total count first
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE source_id = ?1",
            params![source_id],
            |row| row.get(0),
        )?;

        // Build query with optional limit/offset
        let query = match (limit, offset) {
            (Some(l), Some(o)) => format!(
                "SELECT id, source_id, title, file_path, LENGTH(content), created_at
                 FROM documents WHERE source_id = ?1
                 ORDER BY created_at DESC LIMIT {} OFFSET {}",
                l, o
            ),
            (Some(l), None) => format!(
                "SELECT id, source_id, title, file_path, LENGTH(content), created_at
                 FROM documents WHERE source_id = ?1
                 ORDER BY created_at DESC LIMIT {}",
                l
            ),
            _ => "SELECT id, source_id, title, file_path, LENGTH(content), created_at
                  FROM documents WHERE source_id = ?1
                  ORDER BY created_at DESC".to_string(),
        };

        let mut stmt = self.conn.prepare(&query)?;
        let rows = stmt.query_map(params![source_id], |row| {
            Ok(DocumentListItem {
                id: row.get(0)?,
                source_id: row.get(1)?,
                title: row.get(2)?,
                file_path: row.get(3)?,
                content_length: row.get::<_, i64>(4)? as usize,
                created_at: row.get(5)?,
            })
        })?;

        let mut docs = Vec::new();
        for row in rows {
            docs.push(row?);
        }

        Ok((docs, total as usize))
    }

    /// List all sources with stats (for web UI).
    pub fn list_sources(&self) -> Result<Vec<SourceStats>> {
        let mut stmt = self.conn.prepare(
            "SELECT source_id, COUNT(*), SUM(LENGTH(content)), MAX(created_at)
             FROM documents GROUP BY source_id ORDER BY source_id"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(SourceStats {
                id: row.get(0)?,
                doc_count: row.get::<_, i64>(1)? as u64,
                total_size: row.get::<_, i64>(2)? as u64,
                last_updated: row.get(3)?,
            })
        })?;

        let mut sources = Vec::new();
        for row in rows {
            sources.push(row?);
        }

        Ok(sources)
    }

    /// Get all documents (for export) - legacy format.
    pub fn get_all_documents(&self) -> Result<Vec<(String, String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, content, created_at FROM documents")?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        let mut documents = Vec::new();
        for row in rows {
            let (id, compressed, created_at) = row?;
            let content = decompress(&compressed)?;
            documents.push((id, content, created_at));
        }

        Ok(documents)
    }

    /// Get document count (for progress tracking).
    pub fn document_count(&self) -> Result<u64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .map_err(Into::into)
    }

    /// Get all documents with full metadata (for re-indexing).
    /// Returns documents in chunks to avoid loading everything into memory at once.
    pub fn get_all_documents_with_metadata(&self) -> Result<Vec<DocumentRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_id, title, file_path, content, created_at FROM documents",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Vec<u8>>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?;

        let mut documents = Vec::new();
        for row in rows {
            let (id, source_id, title, file_path, compressed, created_at) = row?;
            let content = decompress(&compressed)?;
            documents.push(DocumentRow {
                id,
                source_id,
                title,
                file_path,
                content,
                created_at,
            });
        }

        Ok(documents)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Chunk Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Store a chunk's content.
    pub fn insert_chunk(&self, id: &str, document_id: &str, content: &str) -> Result<()> {
        let compressed = compress(content)?;

        self.conn.execute(
            "INSERT OR REPLACE INTO chunks (id, document_id, content) VALUES (?1, ?2, ?3)",
            params![id, document_id, compressed],
        )?;

        Ok(())
    }

    /// Store multiple chunks in a transaction (batch insert).
    pub fn insert_chunks(&self, chunks: &[(String, String, String)]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;

        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO chunks (id, document_id, content) VALUES (?1, ?2, ?3)",
            )?;

            for (id, document_id, content) in chunks {
                let compressed = compress(content)?;
                stmt.execute(params![id, document_id, compressed])?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Get a chunk's content by ID.
    pub fn get_chunk(&self, id: &str) -> Result<Option<String>> {
        let result: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT content FROM chunks WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?;

        match result {
            Some(compressed) => Ok(Some(decompress(&compressed)?)),
            None => Ok(None),
        }
    }

    /// Get multiple chunks by IDs (batch fetch for search results).
    pub fn get_chunks(&self, ids: &[&str]) -> Result<Vec<(String, String)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        // Build query with placeholders
        let placeholders: Vec<&str> = ids.iter().map(|_| "?").collect();
        let query = format!(
            "SELECT id, content FROM chunks WHERE id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = self.conn.prepare(&query)?;

        // Bind all IDs
        let rows = stmt.query_map(rusqlite::params_from_iter(ids.iter()), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;

        let mut results = Vec::new();
        for row in rows {
            let (id, compressed) = row?;
            let content = decompress(&compressed)?;
            results.push((id, content));
        }

        Ok(results)
    }

    /// Delete all chunks for a document.
    pub fn delete_chunks_for_document(&self, document_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM chunks WHERE document_id = ?1",
            params![document_id],
        )?;

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Source Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Delete all content for a source (by document IDs).
    pub fn delete_source(&self, document_ids: &[&str]) -> Result<()> {
        if document_ids.is_empty() {
            return Ok(());
        }

        let placeholders: Vec<&str> = document_ids.iter().map(|_| "?").collect();

        // Chunks deleted via CASCADE
        let query = format!(
            "DELETE FROM documents WHERE id IN ({})",
            placeholders.join(",")
        );

        self.conn
            .execute(&query, rusqlite::params_from_iter(document_ids.iter()))?;

        Ok(())
    }

    /// Delete all content for a source directly by source_id.
    pub fn delete_source_by_source_id(&self, source_id: &str) -> Result<usize> {
        // Chunks deleted via CASCADE
        let deleted = self
            .conn
            .execute("DELETE FROM documents WHERE source_id = ?1", params![source_id])?;

        Ok(deleted)
    }

    /// Reset all content (delete everything).
    pub fn reset(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            DELETE FROM chunks;
            DELETE FROM documents;
            VACUUM;
            ",
        )?;

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stats
    // ─────────────────────────────────────────────────────────────────────────

    /// Get storage statistics.
    pub fn stats(&self) -> Result<ContentStats> {
        let document_count: u64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;

        let chunk_count: u64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

        let db_size: u64 = self.conn.query_row(
            "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()",
            [],
            |row| row.get(0),
        )?;

        Ok(ContentStats {
            document_count,
            chunk_count,
            db_size_bytes: db_size,
        })
    }
}

/// Storage statistics.
#[derive(Debug, Clone)]
pub struct ContentStats {
    pub document_count: u64,
    pub chunk_count: u64,
    pub db_size_bytes: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Compression Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compress a string using zstd.
fn compress(data: &str) -> Result<Vec<u8>> {
    zstd::encode_all(data.as_bytes(), COMPRESSION_LEVEL)
        .context("Failed to compress content")
}

/// Decompress zstd-compressed data to a string.
fn decompress(data: &[u8]) -> Result<String> {
    let decompressed = zstd::decode_all(data).context("Failed to decompress content")?;
    String::from_utf8(decompressed).context("Decompressed content is not valid UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_document_roundtrip() {
        let dir = tempdir().unwrap();
        let store = ContentStore::open(&dir.path().join("content.db")).unwrap();

        store
            .insert_document(
                "doc1",
                "test-source",
                "Test Doc",
                Some("/path/to/doc.md"),
                "Hello, world!",
                "2024-01-01T00:00:00Z",
            )
            .unwrap();

        let content = store.get_document("doc1").unwrap();
        assert_eq!(content, Some("Hello, world!".to_string()));
    }

    #[test]
    fn test_chunk_batch_insert() {
        let dir = tempdir().unwrap();
        let store = ContentStore::open(&dir.path().join("content.db")).unwrap();

        store
            .insert_document(
                "doc1",
                "test-source",
                "Test Doc",
                None,
                "Full document",
                "2024-01-01T00:00:00Z",
            )
            .unwrap();

        let chunks = vec![
            ("c1".to_string(), "doc1".to_string(), "Chunk 1".to_string()),
            ("c2".to_string(), "doc1".to_string(), "Chunk 2".to_string()),
            ("c3".to_string(), "doc1".to_string(), "Chunk 3".to_string()),
        ];

        store.insert_chunks(&chunks).unwrap();

        let results = store.get_chunks(&["c1", "c3"]).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_cascade_delete() {
        let dir = tempdir().unwrap();
        let store = ContentStore::open(&dir.path().join("content.db")).unwrap();

        store
            .insert_document(
                "doc1",
                "test-source",
                "Test Doc",
                None,
                "Content",
                "2024-01-01T00:00:00Z",
            )
            .unwrap();
        store.insert_chunk("c1", "doc1", "Chunk").unwrap();

        store.delete_document("doc1").unwrap();

        assert!(store.get_document("doc1").unwrap().is_none());
        assert!(store.get_chunk("c1").unwrap().is_none());
    }

    #[test]
    fn test_get_all_documents_with_metadata() {
        let dir = tempdir().unwrap();
        let store = ContentStore::open(&dir.path().join("content.db")).unwrap();

        store
            .insert_document(
                "doc1",
                "source-a",
                "Doc One",
                Some("/path/one.md"),
                "Content one",
                "2024-01-01T00:00:00Z",
            )
            .unwrap();

        store
            .insert_document(
                "doc2",
                "source-b",
                "Doc Two",
                None,
                "Content two",
                "2024-01-02T00:00:00Z",
            )
            .unwrap();

        let docs = store.get_all_documents_with_metadata().unwrap();
        assert_eq!(docs.len(), 2);

        let doc1 = docs.iter().find(|d| d.id == "doc1").unwrap();
        assert_eq!(doc1.source_id, "source-a");
        assert_eq!(doc1.title, "Doc One");
        assert_eq!(doc1.file_path, Some("/path/one.md".to_string()));
        assert_eq!(doc1.content, "Content one");

        let doc2 = docs.iter().find(|d| d.id == "doc2").unwrap();
        assert_eq!(doc2.source_id, "source-b");
        assert_eq!(doc2.file_path, None);
    }

    #[test]
    fn test_compression() {
        let original = "Hello ".repeat(1000); // Repetitive content compresses well
        let compressed = compress(&original).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(original, decompressed);
        assert!(compressed.len() < original.len()); // Should be smaller
    }
}

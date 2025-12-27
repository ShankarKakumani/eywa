//! Batch Writer for atomic writes to LanceDB, SQLite, and Tantivy
//!
//! Writes documents and chunks in large batches to avoid LanceDB fragmentation.
//! Ensures atomicity by writing to SQLite first (content), then LanceDB (vectors),
//! then Tantivy (BM25 index).

use super::{ChunkData, PreparedDoc};
use crate::bm25::{BM25Index, ChunkInput};
use crate::content::ContentStore;
use crate::db::{ChunkRecord, VectorDB};
use crate::types::DocumentRecord;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Statistics from a batch write operation
#[derive(Debug, Default)]
pub struct WriteStats {
    /// Number of documents written
    pub documents_written: u32,
    /// Number of chunks written
    pub chunks_written: u32,
    /// IDs of documents written
    pub document_ids: Vec<String>,
}

impl WriteStats {
    /// Merge another WriteStats into this one
    pub fn merge(&mut self, other: WriteStats) {
        self.documents_written += other.documents_written;
        self.chunks_written += other.chunks_written;
        self.document_ids.extend(other.document_ids);
    }
}

/// Writes batches of documents and chunks to storage
pub struct BatchWriter {
    /// Path to the content database
    content_db_path: PathBuf,
    /// BM25 index for keyword search
    bm25_index: Arc<BM25Index>,
}

impl BatchWriter {
    /// Create a new batch writer for the given data directory
    pub fn new(data_dir: &Path, bm25_index: Arc<BM25Index>) -> Result<Self> {
        Ok(Self {
            content_db_path: data_dir.join("content.db"),
            bm25_index,
        })
    }

    /// Write a batch of documents and chunks to storage
    ///
    /// Order of operations:
    /// 1. Write content to SQLite (content store)
    /// 2. Write document metadata to LanceDB
    /// 3. Write chunk vectors to LanceDB
    /// 4. Index chunks in Tantivy for BM25 search
    ///
    /// This order ensures that if LanceDB write fails, content is still recoverable.
    pub async fn write_batch(
        &mut self,
        db: &mut VectorDB,
        source_id: &str,
        documents: Vec<PreparedDoc>,
        chunks: &[ChunkData],
        embeddings: &[Vec<f32>],
    ) -> Result<WriteStats> {
        if documents.is_empty() {
            return Ok(WriteStats::default());
        }

        let mut stats = WriteStats::default();

        // Phase 1: Write content to SQLite (in a block to drop ContentStore before await)
        {
            let content_store = ContentStore::open(&self.content_db_path)?;

            for doc in &documents {
                // Insert document content with full metadata
                content_store.insert_document(
                    &doc.id,
                    source_id,
                    &doc.title,
                    doc.file_path.as_deref(),
                    &doc.content,
                    &doc.created_at,
                )?;

                // Collect chunk contents for this document
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

        // Phase 2: Write document metadata to LanceDB
        for doc in &documents {
            let doc_record = DocumentRecord {
                id: doc.id.clone(),
                source_id: source_id.to_string(),
                title: doc.title.clone(),
                file_path: doc.file_path.clone(),
                created_at: doc.created_at.clone(),
                chunk_count: doc.chunks.len() as u32,
                content_length: doc.content_length,
            };
            db.insert_document(&doc_record).await?;
            stats.documents_written += 1;
            stats.document_ids.push(doc.id.clone());
        }

        // Phase 3: Write chunk vectors to LanceDB in large batches
        if !chunks.is_empty() && !embeddings.is_empty() {
            let chunk_records: Vec<ChunkRecord> = chunks
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
                    // Preserve hierarchical metadata from smart chunking
                    section: c.section.clone(),
                    subsection: c.subsection.clone(),
                    hierarchy: c.hierarchy.clone(),
                    has_code: c.has_code,
                })
                .collect();

            // Write all chunks in one batch to avoid fragmentation
            db.insert_chunks(&chunk_records, embeddings).await?;
            stats.chunks_written = chunks.len() as u32;

            // Phase 4: Index chunks in Tantivy for BM25 search
            let chunk_inputs: Vec<ChunkInput> = chunks
                .iter()
                .map(|c| ChunkInput {
                    id: c.id.clone(),
                    source_id: c.source_id.clone(),
                    content: c.content.clone(),
                    title: c.title.clone(),
                })
                .collect();

            self.bm25_index.index_chunks(&chunk_inputs)?;
        }

        Ok(stats)
    }
}

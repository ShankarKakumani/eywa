//! LanceDB vector database for storing and searching embeddings
//!
//! Stores only metadata and vectors. Content lives in SQLite (see content.rs).
//! This separation enables efficient storage while maintaining fast vector search.

use crate::config::Config;
use anyhow::{Context, Result};
use arrow_array::{
    Array, BooleanArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection, DistanceType, Table};
use std::sync::Arc;

use crate::types::{ChunkMeta, DocumentMeta, DocumentRecord, Source};

const CHUNKS_TABLE: &str = "chunks_v2";
const DOCS_TABLE: &str = "documents_v2";

/// Maximum limit for queries when all documents are needed.
/// LanceDB v0.15 defaults to 10 if no limit is specified.
pub const MAX_QUERY_LIMIT: usize = 1_000_000;

/// Chunk metadata for insertion (content stored separately in SQLite)
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub id: String,
    pub document_id: String,
    pub source_id: String,
    pub title: Option<String>,
    pub file_path: Option<String>,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub content_hash: String,
    // Hierarchical metadata from smart chunking
    pub section: Option<String>,
    pub subsection: Option<String>,
    pub hierarchy: Vec<String>,
    pub has_code: bool,
}

/// Escape single quotes in strings to prevent SQL injection
fn escape_sql(s: &str) -> String {
    s.replace('\'', "''")
}

pub struct VectorDB {
    conn: Connection,
    chunks_table: Option<Table>,
    docs_table: Option<Table>,
    embedding_dim: usize,
}

impl VectorDB {
    /// Create a new VectorDB instance
    pub async fn new(data_dir: &str) -> Result<Self> {
        // Get embedding dimension from config
        let embedding_dim = Config::load()?
            .map(|c| c.embedding_model.dimensions())
            .unwrap_or(768); // Default to BGE base dimensions

        let conn = connect(data_dir)
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        let chunks_table = conn.open_table(CHUNKS_TABLE).execute().await.ok();
        let docs_table = conn.open_table(DOCS_TABLE).execute().await.ok();

        Ok(Self {
            conn,
            chunks_table,
            docs_table,
            embedding_dim,
        })
    }

    /// Get or create the chunks table
    async fn get_or_create_chunks_table(&mut self) -> Result<Table> {
        if let Some(ref table) = self.chunks_table {
            return Ok(table.clone());
        }

        let schema = self.chunks_schema();
        let table = self
            .conn
            .create_empty_table(CHUNKS_TABLE, schema)
            .execute()
            .await
            .context("Failed to create chunks table")?;

        self.chunks_table = Some(table.clone());
        Ok(table)
    }

    /// Get or create the documents table
    async fn get_or_create_docs_table(&mut self) -> Result<Table> {
        if let Some(ref table) = self.docs_table {
            return Ok(table.clone());
        }

        let schema = Self::docs_schema();
        let table = self
            .conn
            .create_empty_table(DOCS_TABLE, schema)
            .execute()
            .await
            .context("Failed to create documents table")?;

        self.docs_table = Some(table.clone());
        Ok(table)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Schema Definitions (NO content fields - content lives in SQLite)
    // ─────────────────────────────────────────────────────────────────────────

    /// Schema for documents table (metadata only, no content)
    fn docs_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("source_id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, true),
            Field::new("created_at", DataType::Utf8, false),
            Field::new("chunk_count", DataType::UInt32, false),
            Field::new("content_length", DataType::UInt32, false),
        ]))
    }

    /// Schema for chunks table (metadata + vectors, no content)
    fn chunks_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document_id", DataType::Utf8, false),
            Field::new("source_id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, true),
            Field::new("file_path", DataType::Utf8, true),
            Field::new("line_start", DataType::UInt32, true),
            Field::new("line_end", DataType::UInt32, true),
            Field::new("content_hash", DataType::Utf8, false),
            // Hierarchical metadata from smart chunking
            Field::new("section", DataType::Utf8, true),
            Field::new("subsection", DataType::Utf8, true),
            Field::new("hierarchy", DataType::Utf8, true), // JSON serialized
            Field::new("has_code", DataType::Boolean, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.embedding_dim as i32,
                ),
                false,
            ),
        ]))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Document Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Insert a document record (metadata only, content stored in SQLite)
    pub async fn insert_document(&mut self, doc: &DocumentRecord) -> Result<()> {
        let table = self.get_or_create_docs_table().await?;

        let schema = Self::docs_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![doc.id.as_str()])),
                Arc::new(StringArray::from(vec![doc.source_id.as_str()])),
                Arc::new(StringArray::from(vec![doc.title.as_str()])),
                Arc::new(StringArray::from(vec![doc.file_path.as_deref()])),
                Arc::new(StringArray::from(vec![doc.created_at.as_str()])),
                Arc::new(UInt32Array::from(vec![doc.chunk_count])),
                Arc::new(UInt32Array::from(vec![doc.content_length])),
            ],
        )?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table.add(batches).execute().await?;

        Ok(())
    }

    /// Get a document record by ID (metadata only)
    pub async fn get_document(&self, doc_id: &str) -> Result<Option<DocumentRecord>> {
        let table = match &self.docs_table {
            Some(t) => t,
            None => return Ok(None),
        };

        let results = table
            .query()
            .only_if(format!("id = '{}'", escape_sql(doc_id)))
            .limit(1)
            .execute()
            .await?;

        let batches: Vec<RecordBatch> = results.try_collect().await?;

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            return Ok(Self::extract_document_record(&batch, 0));
        }

        Ok(None)
    }

    /// List all documents in a source
    /// Note: LanceDB v0.15 defaults to limit=10, so we explicitly set a limit.
    /// Pass None for default (10), or Some(n) for custom limit.
    pub async fn list_documents(&self, source_id: &str, limit: Option<usize>) -> Result<Vec<DocumentMeta>> {
        let table = match &self.docs_table {
            Some(t) => t,
            None => return Ok(vec![]),
        };

        let results = table
            .query()
            .only_if(format!("source_id = '{}'", escape_sql(source_id)))
            .limit(limit.unwrap_or(10))
            .execute()
            .await?;

        let batches: Vec<RecordBatch> = results.try_collect().await?;
        let mut docs = Vec::new();

        for batch in batches {
            for i in 0..batch.num_rows() {
                if let Some(record) = Self::extract_document_record(&batch, i) {
                    docs.push(DocumentMeta {
                        id: record.id,
                        source_id: record.source_id,
                        title: record.title,
                        file_path: record.file_path,
                        created_at: record.created_at,
                        chunk_count: record.chunk_count,
                        content_length: record.content_length as usize,
                    });
                }
            }
        }

        Ok(docs)
    }

    /// Get all document records (for export)
    /// Note: LanceDB v0.15 defaults to limit=10, so we explicitly set a limit.
    /// Pass None for default (10), or Some(n) for custom limit.
    pub async fn get_all_document_records(&self, limit: Option<usize>) -> Result<Vec<DocumentRecord>> {
        let table = match &self.docs_table {
            Some(t) => t,
            None => return Ok(vec![]),
        };

        let results = table.query().limit(limit.unwrap_or(10)).execute().await?;
        let batches: Vec<RecordBatch> = results.try_collect().await?;
        let mut docs = Vec::new();

        for batch in batches {
            for i in 0..batch.num_rows() {
                if let Some(record) = Self::extract_document_record(&batch, i) {
                    docs.push(record);
                }
            }
        }

        Ok(docs)
    }

    /// Get document IDs for a source (for deletion)
    /// Uses high limit to ensure all docs are returned (internal use).
    pub async fn get_document_ids_for_source(&self, source_id: &str) -> Result<Vec<String>> {
        let table = match &self.docs_table {
            Some(t) => t,
            None => return Ok(vec![]),
        };

        let results = table
            .query()
            .only_if(format!("source_id = '{}'", escape_sql(source_id)))
            .limit(MAX_QUERY_LIMIT) // Need all docs for deletion
            .execute()
            .await?;

        let batches: Vec<RecordBatch> = results.try_collect().await?;
        let mut ids = Vec::new();

        for batch in batches {
            if let Some(id_col) = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            {
                for i in 0..batch.num_rows() {
                    ids.push(id_col.value(i).to_string());
                }
            }
        }

        Ok(ids)
    }

    /// Extract a document record from a batch row
    fn extract_document_record(batch: &RecordBatch, idx: usize) -> Option<DocumentRecord> {
        let ids = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())?;
        let source_ids = batch
            .column_by_name("source_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())?;
        let titles = batch
            .column_by_name("title")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())?;
        let file_paths = batch
            .column_by_name("file_path")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let created_ats = batch
            .column_by_name("created_at")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())?;
        let chunk_counts = batch
            .column_by_name("chunk_count")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())?;
        let content_lengths = batch
            .column_by_name("content_length")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())?;

        Some(DocumentRecord {
            id: ids.value(idx).to_string(),
            source_id: source_ids.value(idx).to_string(),
            title: titles.value(idx).to_string(),
            file_path: file_paths.and_then(|f| {
                if f.is_null(idx) {
                    None
                } else {
                    Some(f.value(idx).to_string())
                }
            }),
            created_at: created_ats.value(idx).to_string(),
            chunk_count: chunk_counts.value(idx),
            content_length: content_lengths.value(idx),
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Chunk Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Insert chunks with their embeddings (metadata only, content in SQLite)
    pub async fn insert_chunks(
        &mut self,
        chunks: &[ChunkRecord],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let table = self.get_or_create_chunks_table().await?;

        let ids: Vec<&str> = chunks.iter().map(|c| c.id.as_str()).collect();
        let document_ids: Vec<&str> = chunks.iter().map(|c| c.document_id.as_str()).collect();
        let source_ids: Vec<&str> = chunks.iter().map(|c| c.source_id.as_str()).collect();
        let titles: Vec<Option<&str>> = chunks.iter().map(|c| c.title.as_deref()).collect();
        let file_paths: Vec<Option<&str>> = chunks.iter().map(|c| c.file_path.as_deref()).collect();
        let line_starts: Vec<Option<u32>> = chunks.iter().map(|c| c.line_start).collect();
        let line_ends: Vec<Option<u32>> = chunks.iter().map(|c| c.line_end).collect();
        let content_hashes: Vec<&str> = chunks.iter().map(|c| c.content_hash.as_str()).collect();

        // Hierarchical metadata from smart chunking
        let sections: Vec<Option<&str>> = chunks.iter().map(|c| c.section.as_deref()).collect();
        let subsections: Vec<Option<&str>> =
            chunks.iter().map(|c| c.subsection.as_deref()).collect();
        let hierarchies: Vec<String> = chunks
            .iter()
            .map(|c| serde_json::to_string(&c.hierarchy).unwrap_or_else(|_| "[]".to_string()))
            .collect();
        let hierarchy_refs: Vec<&str> = hierarchies.iter().map(|s| s.as_str()).collect();
        let has_codes: Vec<bool> = chunks.iter().map(|c| c.has_code).collect();

        let flat_embeddings: Vec<f32> = embeddings.iter().flatten().copied().collect();

        let schema = self.chunks_schema();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(document_ids)),
                Arc::new(StringArray::from(source_ids)),
                Arc::new(StringArray::from(titles)),
                Arc::new(StringArray::from(file_paths)),
                Arc::new(UInt32Array::from(line_starts)),
                Arc::new(UInt32Array::from(line_ends)),
                Arc::new(StringArray::from(content_hashes)),
                // Hierarchical metadata
                Arc::new(StringArray::from(sections)),
                Arc::new(StringArray::from(subsections)),
                Arc::new(StringArray::from(hierarchy_refs)),
                Arc::new(BooleanArray::from(has_codes)),
                Arc::new(arrow_array::FixedSizeListArray::new(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.embedding_dim as i32,
                    Arc::new(Float32Array::from(flat_embeddings)),
                    None,
                )),
            ],
        )?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table.add(batches).execute().await?;

        Ok(())
    }

    /// Search for similar chunks (returns metadata only, content from SQLite)
    pub async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<ChunkMeta>> {
        self.search_filtered(query_embedding, limit, None).await
    }

    /// Search for similar chunks with optional source filter
    pub async fn search_filtered(
        &self,
        query_embedding: &[f32],
        limit: usize,
        source_id: Option<&str>,
    ) -> Result<Vec<ChunkMeta>> {
        let table = match &self.chunks_table {
            Some(t) => t,
            None => return Ok(vec![]),
        };

        let mut query = table
            .vector_search(query_embedding.to_vec())
            .context("Failed to create vector search")?
            .distance_type(DistanceType::Cosine)
            .limit(limit);

        if let Some(source) = source_id {
            query = query.only_if(format!("source_id = '{}'", escape_sql(source)));
        }

        let results = query
            .execute()
            .await
            .context("Failed to execute search")?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .context("Failed to collect results")?;

        let mut search_results = Vec::new();

        for batch in batches {
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let document_ids = batch
                .column_by_name("document_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let source_ids = batch
                .column_by_name("source_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let titles = batch
                .column_by_name("title")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let file_paths = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let line_starts = batch
                .column_by_name("line_start")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let line_ends = batch
                .column_by_name("line_end")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            if let (Some(ids), Some(document_ids), Some(source_ids), Some(distances)) =
                (ids, document_ids, source_ids, distances)
            {
                for i in 0..batch.num_rows() {
                    let score = 1.0 - distances.value(i);
                    search_results.push(ChunkMeta {
                        id: ids.value(i).to_string(),
                        document_id: document_ids.value(i).to_string(),
                        source_id: source_ids.value(i).to_string(),
                        title: titles.and_then(|t| {
                            if t.is_null(i) {
                                None
                            } else {
                                Some(t.value(i).to_string())
                            }
                        }),
                        file_path: file_paths.and_then(|f| {
                            if f.is_null(i) {
                                None
                            } else {
                                Some(f.value(i).to_string())
                            }
                        }),
                        line_start: line_starts.and_then(|l| {
                            if l.is_null(i) {
                                None
                            } else {
                                Some(l.value(i))
                            }
                        }),
                        line_end: line_ends.and_then(|l| {
                            if l.is_null(i) {
                                None
                            } else {
                                Some(l.value(i))
                            }
                        }),
                        score,
                    });
                }
            }
        }

        Ok(search_results)
    }

    /// Check if a chunk already exists by content hash
    pub async fn chunk_exists(&self, content_hash: &str) -> Result<bool> {
        let table = match &self.chunks_table {
            Some(t) => t,
            None => return Ok(false),
        };

        let results = table
            .query()
            .only_if(format!("content_hash = '{}'", escape_sql(content_hash)))
            .limit(1)
            .execute()
            .await?;

        let batches: Vec<RecordBatch> = results.try_collect().await?;
        Ok(batches.iter().any(|b| b.num_rows() > 0))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Source Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// List all sources with aggregated statistics.
    /// Uses high limit to ensure all docs are scanned for aggregation.
    pub async fn list_sources(&self) -> Result<Vec<Source>> {
        let table = match &self.docs_table {
            Some(t) => t,
            None => return Ok(vec![]),
        };

        let results = table.query().limit(MAX_QUERY_LIMIT).execute().await?;
        let batches: Vec<RecordBatch> = results.try_collect().await?;

        // Track chunk counts, unique document IDs, and latest created_at per source
        let mut source_chunks: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();
        let mut source_docs: std::collections::HashMap<String, std::collections::HashSet<String>> =
            std::collections::HashMap::new();
        let mut source_latest: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        for batch in batches {
            let source_ids = batch
                .column_by_name("source_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let chunk_counts = batch
                .column_by_name("chunk_count")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let doc_ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let created_ats = batch
                .column_by_name("created_at")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            if let (Some(source_ids), Some(chunk_counts)) = (source_ids, chunk_counts) {
                for i in 0..batch.num_rows() {
                    let source_id = source_ids.value(i).to_string();
                    let chunk_count = chunk_counts.value(i) as u64;
                    *source_chunks.entry(source_id.clone()).or_insert(0) += chunk_count;

                    // Track unique document IDs
                    if let Some(doc_ids) = doc_ids {
                        let doc_id = doc_ids.value(i).to_string();
                        source_docs
                            .entry(source_id.clone())
                            .or_insert_with(std::collections::HashSet::new)
                            .insert(doc_id);
                    }

                    // Track latest created_at per source
                    if let Some(created_ats) = created_ats {
                        let date = created_ats.value(i).to_string();
                        source_latest
                            .entry(source_id)
                            .and_modify(|existing| {
                                if date > *existing {
                                    *existing = date.clone();
                                }
                            })
                            .or_insert(date);
                    }
                }
            }
        }

        let sources: Vec<Source> = source_chunks
            .into_iter()
            .map(|(id, chunk_count)| {
                let doc_count = source_docs.get(&id).map(|s| s.len() as u64).unwrap_or(0);
                let last_indexed = source_latest.get(&id).cloned();
                Source {
                    id: id.clone(),
                    name: id,
                    description: None,
                    doc_count,
                    chunk_count,
                    last_indexed,
                }
            })
            .collect();

        Ok(sources)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Deletion Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Delete a document and its chunks from LanceDB
    pub async fn delete_document(&self, doc_id: &str) -> Result<()> {
        let escaped_id = escape_sql(doc_id);

        if let Some(ref table) = self.docs_table {
            table.delete(&format!("id = '{}'", escaped_id)).await?;
        }

        if let Some(ref table) = self.chunks_table {
            table
                .delete(&format!("document_id = '{}'", escaped_id))
                .await?;
        }

        Ok(())
    }

    /// Delete all documents and chunks for a source
    pub async fn delete_source(&self, source_id: &str) -> Result<()> {
        let escaped_id = escape_sql(source_id);

        if let Some(ref table) = self.chunks_table {
            table
                .delete(&format!("source_id = '{}'", escaped_id))
                .await?;
        }
        if let Some(ref table) = self.docs_table {
            table
                .delete(&format!("source_id = '{}'", escaped_id))
                .await?;
        }

        Ok(())
    }

    /// Reset everything - delete all data
    pub async fn reset_all(&mut self) -> Result<()> {
        if self.chunks_table.is_some() {
            self.conn.drop_table(CHUNKS_TABLE).await.ok();
            self.chunks_table = None;
        }
        if self.docs_table.is_some() {
            self.conn.drop_table(DOCS_TABLE).await.ok();
            self.docs_table = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_sql() {
        assert_eq!(escape_sql("normal"), "normal");
        assert_eq!(escape_sql("it's"), "it''s");
        assert_eq!(escape_sql("a'b'c"), "a''b''c");
        assert_eq!(escape_sql(""), "");
    }

    #[test]
    fn test_escape_sql_no_quotes() {
        assert_eq!(escape_sql("hello world"), "hello world");
        assert_eq!(escape_sql("source-name_123"), "source-name_123");
    }

    #[test]
    fn test_escape_sql_multiple_quotes() {
        assert_eq!(escape_sql("it's John's"), "it''s John''s");
    }
}

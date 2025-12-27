//! Batch Ingestion Pipeline for Eywa
//!
//! Solves LanceDB fragmentation by accumulating documents and chunks
//! before writing them in large batches.
//!
//! Architecture:
//! - IngestPipeline: Coordinates the ingestion flow
//! - BatchAccumulator: Holds documents until threshold reached
//! - BatchWriter: Writes batches atomically to LanceDB + SQLite
//! - ProgressTracker: Tracks and displays ingestion progress

pub mod accumulator;
pub mod progress;
pub mod writer;

pub use accumulator::BatchAccumulator;
pub use progress::ProgressTracker;
pub use writer::{BatchWriter, WriteStats};

use crate::bm25::BM25Index;
use crate::chunking::{ChunkerRegistry, DocMetadata};
use crate::db::VectorDB;
use crate::embed::Embedder;
use crate::types::{DocumentInput, IngestResponse};
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use walkdir::WalkDir;

/// Configuration for batch ingestion thresholds
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum documents per batch before flush
    pub max_docs: usize,
    /// Maximum chunks per LanceDB write
    pub max_chunks: usize,
    /// Maximum memory in MB before flush
    pub max_memory_mb: usize,
    /// Flush timeout in seconds for partial batches
    pub flush_timeout_secs: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_docs: 50,
            max_chunks: 5000,
            max_memory_mb: 100,
            flush_timeout_secs: 5,
        }
    }
}

/// Batch size for embedding generation
const EMBEDDING_BATCH_SIZE: usize = 32;

/// Prepared document with its chunks ready for processing
#[derive(Debug, Clone)]
pub struct PreparedDoc {
    pub id: String,
    pub content: String,
    pub title: String,
    pub file_path: Option<String>,
    pub created_at: String,
    pub content_length: u32,
    pub chunks: Vec<ChunkData>,
}

/// Intermediate chunk structure during ingestion
#[derive(Debug, Clone)]
pub struct ChunkData {
    pub id: String,
    pub document_id: String,
    pub source_id: String,
    pub title: Option<String>,
    pub content: String,
    pub file_path: Option<String>,
    pub line_start: u32,
    pub line_end: u32,
    pub content_hash: String,
    // Hierarchical metadata from smart chunking
    pub section: Option<String>,
    pub subsection: Option<String>,
    pub hierarchy: Vec<String>,
    pub has_code: bool,
}

/// Batch of documents with embeddings, ready to write to DB
#[derive(Debug)]
pub struct EmbeddedBatch {
    pub source_id: String,
    pub data_dir: std::path::PathBuf,
    pub documents: Vec<PreparedDoc>,
    pub chunks: Vec<ChunkData>,
    pub embeddings: Vec<Vec<f32>>,
}

/// Ingestion pipeline that accumulates and batch-writes documents
pub struct IngestPipeline {
    config: BatchConfig,
    embedder: Arc<Embedder>,
    bm25_index: Arc<BM25Index>,
    chunker: ChunkerRegistry,
}

impl IngestPipeline {
    /// Create a new ingestion pipeline
    pub fn new(embedder: Arc<Embedder>, bm25_index: Arc<BM25Index>) -> Self {
        Self::with_config(embedder, bm25_index, BatchConfig::default())
    }

    /// Create a new ingestion pipeline with custom config
    pub fn with_config(embedder: Arc<Embedder>, bm25_index: Arc<BM25Index>, config: BatchConfig) -> Self {
        Self {
            config,
            embedder,
            bm25_index,
            chunker: ChunkerRegistry::new(),
        }
    }

    /// Check if file extension is supported for ingestion
    fn is_supported_extension(ext: &str) -> bool {
        matches!(
            ext,
            "md" | "txt" | "pdf"
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

    /// Get current ISO timestamp
    fn now_iso() -> String {
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
    }

    /// Prepare a document for ingestion (parse and chunk using content-aware chunking)
    fn prepare_document(
        &self,
        doc_input: &DocumentInput,
        source_id: &str,
    ) -> Option<PreparedDoc> {
        if doc_input.content.trim().is_empty() {
            return None;
        }

        let doc_id = uuid::Uuid::new_v4().to_string();
        let title = doc_input
            .title
            .clone()
            .unwrap_or_else(|| format!("Untitled-{}", &doc_id[..8]));
        let created_at = Self::now_iso();
        let content_length = doc_input.content.len() as u32;

        // Use content-aware chunking based on file type
        let doc_metadata = DocMetadata {
            document_id: doc_id.clone(),
            source_id: source_id.to_string(),
            file_path: doc_input.file_path.clone(),
        };

        let raw_chunks = self.chunker.chunk(
            &doc_input.content,
            doc_input.file_path.as_deref(),
            &doc_metadata,
        );

        // Convert chunking::Chunk to pipeline::ChunkData (preserve all metadata!)
        let chunks: Vec<ChunkData> = raw_chunks
            .into_iter()
            .map(|c| ChunkData {
                id: c.id,
                document_id: doc_id.clone(),
                source_id: source_id.to_string(),
                title: c.metadata.title,
                content: c.content,
                file_path: c.metadata.file_path,
                line_start: c.metadata.line_start,
                line_end: c.metadata.line_end,
                content_hash: c.metadata.content_hash,
                // Preserve hierarchical metadata from smart chunking
                section: c.metadata.section,
                subsection: c.metadata.subsection,
                hierarchy: c.metadata.hierarchy,
                has_code: c.metadata.has_code,
            })
            .collect();

        Some(PreparedDoc {
            id: doc_id,
            content: doc_input.content.clone(),
            title,
            file_path: doc_input.file_path.clone(),
            created_at,
            content_length,
            chunks,
        })
    }

    /// Ingest documents with batched writes to LanceDB
    ///
    /// This is the main entry point for the pipeline. It:
    /// 1. Prepares all documents (chunking)
    /// 2. Accumulates until batch threshold, then flushes
    /// 3. Repeats until all documents processed
    pub async fn ingest_documents(
        &self,
        db: &mut VectorDB,
        data_dir: &Path,
        source_id: &str,
        documents: Vec<DocumentInput>,
    ) -> Result<IngestResponse> {
        let mut accumulator = BatchAccumulator::new(self.config.clone());
        let mut writer = BatchWriter::new(data_dir, Arc::clone(&self.bm25_index))?;
        let mut total_stats = WriteStats::default();
        let mut total_skipped = 0u32;
        let mut batch_num = 0usize;

        // Use ProgressTracker for consistent progress reporting
        let mut progress = ProgressTracker::new(documents.len());

        // Phase 1: Prepare all documents (cheap - just parsing and chunking)
        progress.start_phase(&format!("Preparing {} documents", documents.len()));
        let prepared_docs: Vec<PreparedDoc> = documents
            .iter()
            .filter_map(|doc| self.prepare_document(doc, source_id))
            .collect();
        progress.finish_phase();

        if prepared_docs.is_empty() {
            return Ok(IngestResponse {
                source_id: source_id.to_string(),
                documents_created: 0,
                chunks_created: 0,
                chunks_skipped: 0,
                document_ids: vec![],
            });
        }

        // Phase 2: Process with batch flushing
        for doc in prepared_docs {
            let should_flush = accumulator.add_document(doc);

            if should_flush {
                batch_num += 1;
                let (stats, skipped) = self
                    .flush_batch(&mut accumulator, &mut writer, db, source_id, batch_num, &mut progress)
                    .await?;
                total_stats.merge(stats);
                total_skipped += skipped;
            }
        }

        // Final flush for remaining documents
        if !accumulator.is_empty() {
            batch_num += 1;
            let (stats, skipped) = self
                .flush_batch(&mut accumulator, &mut writer, db, source_id, batch_num, &mut progress)
                .await?;
            total_stats.merge(stats);
            total_skipped += skipped;
        }

        // Update progress with final counts
        progress.update_docs(total_stats.documents_written as usize);
        progress.update_chunks(total_stats.chunks_written as usize);
        progress.complete();

        Ok(IngestResponse {
            source_id: source_id.to_string(),
            documents_created: total_stats.documents_written,
            chunks_created: total_stats.chunks_written,
            chunks_skipped: total_skipped,
            document_ids: total_stats.document_ids,
        })
    }

    /// Flush a batch: deduplicate, embed, and write to storage
    async fn flush_batch(
        &self,
        accumulator: &mut BatchAccumulator,
        writer: &mut BatchWriter,
        db: &mut VectorDB,
        source_id: &str,
        batch_num: usize,
        progress: &mut ProgressTracker,
    ) -> Result<(WriteStats, u32)> {
        let doc_count = accumulator.document_count();
        let chunk_count = accumulator.chunk_count();
        progress.start_phase(&format!(
            "Batch {}: {} docs, {} chunks",
            batch_num, doc_count, chunk_count
        ));

        // Step 1: Check for duplicate chunks
        let mut chunks_to_embed: Vec<ChunkData> = Vec::new();
        let mut chunks_skipped = 0u32;

        for chunk in accumulator.all_chunks() {
            if db.chunk_exists(&chunk.content_hash).await? {
                chunks_skipped += 1;
            } else {
                chunks_to_embed.push(chunk.clone());
            }
        }

        // Step 2: Generate embeddings
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks_to_embed.len());

        for batch in chunks_to_embed.chunks(EMBEDDING_BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|c| c.content.clone()).collect();
            let embeddings = self.embedder.embed_batch(&texts)?;
            all_embeddings.extend(embeddings);
        }

        // Step 3: Write to storage
        let documents = accumulator.take_documents();
        let stats = writer
            .write_batch(db, source_id, documents, &chunks_to_embed, &all_embeddings)
            .await?;

        progress.finish_phase();
        Ok((stats, chunks_skipped))
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

    /// Prepare documents and generate embeddings WITHOUT needing DB access
    /// Use this to avoid holding DB lock during slow embedding
    pub fn prepare_and_embed(
        &self,
        source_id: &str,
        data_dir: &Path,
        documents: Vec<DocumentInput>,
    ) -> Result<EmbeddedBatch> {
        // Step 1: Prepare all documents (chunking)
        let prepared_docs: Vec<PreparedDoc> = documents
            .iter()
            .filter_map(|doc| self.prepare_document(doc, source_id))
            .collect();

        if prepared_docs.is_empty() {
            return Ok(EmbeddedBatch {
                source_id: source_id.to_string(),
                data_dir: data_dir.to_path_buf(),
                documents: vec![],
                chunks: vec![],
                embeddings: vec![],
            });
        }

        // Step 2: Collect all chunks
        let all_chunks: Vec<ChunkData> = prepared_docs
            .iter()
            .flat_map(|doc| doc.chunks.clone())
            .collect();

        // Step 3: Generate embeddings (the slow part - no lock needed!)
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(all_chunks.len());
        for batch in all_chunks.chunks(EMBEDDING_BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|c| c.content.clone()).collect();
            let embeddings = self.embedder.embed_batch(&texts)?;
            all_embeddings.extend(embeddings);
        }

        Ok(EmbeddedBatch {
            source_id: source_id.to_string(),
            data_dir: data_dir.to_path_buf(),
            documents: prepared_docs,
            chunks: all_chunks,
            embeddings: all_embeddings,
        })
    }

    /// Write a pre-embedded batch to DB - call this with DB lock held (fast operation)
    pub async fn write_embedded_batch(
        &self,
        db: &mut VectorDB,
        batch: EmbeddedBatch,
    ) -> Result<IngestResponse> {
        let mut writer = BatchWriter::new(&batch.data_dir, Arc::clone(&self.bm25_index))?;

        // Filter out duplicate chunks
        let mut chunks_to_write: Vec<&ChunkData> = Vec::new();
        let mut embeddings_to_write: Vec<&Vec<f32>> = Vec::new();
        let mut chunks_skipped = 0u32;

        for (chunk, embedding) in batch.chunks.iter().zip(batch.embeddings.iter()) {
            if db.chunk_exists(&chunk.content_hash).await? {
                chunks_skipped += 1;
            } else {
                chunks_to_write.push(chunk);
                embeddings_to_write.push(embedding);
            }
        }

        // Write to storage
        let chunks_owned: Vec<ChunkData> = chunks_to_write.into_iter().cloned().collect();
        let embeddings_owned: Vec<Vec<f32>> = embeddings_to_write.into_iter().cloned().collect();

        let stats = writer
            .write_batch(
                db,
                &batch.source_id,
                batch.documents,
                &chunks_owned,
                &embeddings_owned,
            )
            .await?;

        Ok(IngestResponse {
            source_id: batch.source_id,
            documents_created: stats.documents_written,
            chunks_created: stats.chunks_written,
            chunks_skipped,
            document_ids: stats.document_ids,
        })
    }
}

//! Eywa - Personal knowledge base with local embeddings and vector search
//!
//! Hybrid storage architecture:
//! - LanceDB: vectors + metadata (fast search)
//! - SQLite: content (compressed, efficient)
//! - Tantivy: BM25 keyword search (hybrid retrieval)

pub mod bm25;
pub mod chunking;
pub mod config;
pub mod content;
pub mod db;
pub mod embed;
pub mod ingest;
pub mod init;
pub mod job;
pub mod pipeline;
pub mod repl;
pub mod rerank;
pub mod search;
pub mod setup;
pub mod types;

pub use bm25::{BM25Index, BM25Result, ChunkInput};
pub use config::{Config, DevicePreference, EmbeddingModel, RerankerModel};
pub use content::{ContentStore, DocumentListItem, DocumentRow, SourceStats};
pub use db::{ChunkRecord, VectorDB};
pub use embed::{gpu_support_info, Embedder, GpuSupportInfo};
pub use ingest::Ingester;
pub use init::{run_init, show_status, show_welcome, InitResult};
pub use job::{create_job_queue, JobQueue, PendingDocInfo, SharedJobQueue};
pub use setup::{run_download_wizard, models_cached};
pub use pipeline::{BatchConfig, EmbeddedBatch, IngestPipeline};
pub use rerank::Reranker;
pub use search::SearchEngine;
pub use types::*;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

/// Eywa knowledge base instance
pub struct Eywa {
    pub embedder: RwLock<Embedder>,
    pub db: RwLock<VectorDB>,
    pub bm25_index: Arc<BM25Index>,
    pub content: Mutex<ContentStore>,
    pub search: SearchEngine,
}

impl Eywa {
    /// Create a new Eywa instance
    pub async fn new(data_dir: &str) -> anyhow::Result<Self> {
        let embedder = Embedder::new()?;
        let db = VectorDB::new(data_dir).await?;
        let bm25_index = Arc::new(BM25Index::open(Path::new(data_dir))?);

        let content_path = Path::new(data_dir).join("content.db");
        let content = ContentStore::open(&content_path)?;

        let search = SearchEngine::new();

        Ok(Self {
            embedder: RwLock::new(embedder),
            db: RwLock::new(db),
            bm25_index,
            content: Mutex::new(content),
            search,
        })
    }

    /// Ingest documents from a file or directory
    pub async fn ingest_path(&self, source_id: &str, path: &str, data_dir: &Path) -> anyhow::Result<IngestResponse> {
        let mut db = self.db.write().await;
        let embedder = self.embedder.read().await;
        let ingester = Ingester::new(&embedder);
        ingester
            .ingest_from_path(&mut db, data_dir, source_id, path)
            .await
    }

    /// Ingest documents from API input
    pub async fn ingest_documents(
        &self,
        source_id: &str,
        documents: Vec<DocumentInput>,
        data_dir: &Path,
    ) -> anyhow::Result<IngestResponse> {
        let mut db = self.db.write().await;
        let embedder = self.embedder.read().await;
        let ingester = Ingester::new(&embedder);
        ingester
            .ingest_documents(&mut db, data_dir, source_id, documents)
            .await
    }

    /// Search for documents using hybrid retrieval (vector + BM25)
    ///
    /// Combines semantic search (vector similarity) with keyword search (BM25)
    /// using convex combination: 0.8 * vector + 0.2 * bm25
    pub async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchResult>> {
        let embedder = self.embedder.read().await;
        let query_embedding = embedder.embed(query)?;
        let db = self.db.read().await;

        // Hybrid search: vector + BM25
        let vector_limit = 50;
        let bm25_limit = 50;

        // Step 1: Get vector search results
        let chunk_metas = db.search(&query_embedding, vector_limit).await?;

        // Step 2: Get BM25 search results
        let bm25_results = self.bm25_index.search(query, bm25_limit)?;

        // Step 3: Normalize and fuse scores
        let fused_scores = Self::convex_fusion(&chunk_metas, &bm25_results, 0.8, 0.2);

        if fused_scores.is_empty() {
            return Ok(vec![]);
        }

        // Take top candidates for content fetch
        let top_ids: Vec<String> = fused_scores
            .iter()
            .take(limit * 2)
            .map(|(id, _)| id.clone())
            .collect();

        // Step 4: Fetch content from SQLite
        let id_refs: Vec<&str> = top_ids.iter().map(|s| s.as_str()).collect();
        let content = self.content.lock().unwrap();
        let contents = content.get_chunks(&id_refs)?;
        let content_map: HashMap<String, String> = contents.into_iter().collect();

        // Build a map of chunk metadata by ID
        let meta_map: HashMap<String, &ChunkMeta> = chunk_metas
            .iter()
            .map(|m| (m.id.clone(), m))
            .collect();

        // Step 5: Combine into SearchResult with fused scores
        let mut results: Vec<SearchResult> = fused_scores
            .iter()
            .take(limit * 2)
            .filter_map(|(id, fused_score)| {
                let content_text = content_map.get(id)?.clone();
                // Try to get metadata from vector results, or create minimal metadata
                if let Some(meta) = meta_map.get(id) {
                    Some(SearchResult {
                        id: meta.id.clone(),
                        source_id: meta.source_id.clone(),
                        title: meta.title.clone(),
                        content: content_text,
                        file_path: meta.file_path.clone(),
                        line_start: meta.line_start,
                        score: *fused_score,
                    })
                } else {
                    // BM25-only result - need to fetch metadata
                    None
                }
            })
            .collect();

        // Filter and rerank
        results = self.search.filter_results(results);
        results = self.search.rerank_with_keywords(results, query);

        Ok(results.into_iter().take(limit).collect())
    }

    /// Normalize scores to [0, 1] range using min-max normalization
    fn normalize_scores(scores: &[(String, f32)]) -> Vec<(String, f32)> {
        if scores.is_empty() {
            return vec![];
        }

        let min = scores.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
        let max = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range == 0.0 {
            // All scores are the same
            return scores.iter().map(|(id, _)| (id.clone(), 1.0)).collect();
        }

        scores
            .iter()
            .map(|(id, score)| (id.clone(), (score - min) / range))
            .collect()
    }

    /// Convex combination of vector and BM25 scores
    fn convex_fusion(
        vector_results: &[ChunkMeta],
        bm25_results: &[BM25Result],
        vec_weight: f32,
        bm25_weight: f32,
    ) -> Vec<(String, f32)> {
        // Extract scores as (id, score) pairs
        let vec_scores: Vec<(String, f32)> = vector_results
            .iter()
            .map(|r| (r.id.clone(), r.score))
            .collect();

        let bm25_scores: Vec<(String, f32)> = bm25_results
            .iter()
            .map(|r| (r.chunk_id.clone(), r.score))
            .collect();

        // Normalize both score sets
        let vec_normalized = Self::normalize_scores(&vec_scores);
        let bm25_normalized = Self::normalize_scores(&bm25_scores);

        // Combine scores using convex combination
        let mut combined: HashMap<String, f32> = HashMap::new();

        for (id, score) in vec_normalized {
            *combined.entry(id).or_default() += vec_weight * score;
        }
        for (id, score) in bm25_normalized {
            *combined.entry(id).or_default() += bm25_weight * score;
        }

        // Sort by combined score descending
        let mut results: Vec<(String, f32)> = combined.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// List all sources
    pub async fn list_sources(&self) -> anyhow::Result<Vec<Source>> {
        let db = self.db.read().await;
        db.list_sources().await
    }

    /// List documents in a source
    /// Pass None for default limit (10), or Some(n) for custom limit.
    pub async fn list_documents(&self, source_id: &str, limit: Option<usize>) -> anyhow::Result<Vec<DocumentMeta>> {
        let db = self.db.read().await;
        db.list_documents(source_id, limit).await
    }

    /// Get a document by ID (metadata from LanceDB, content from SQLite)
    pub async fn get_document(&self, doc_id: &str) -> anyhow::Result<Option<Document>> {
        let db = self.db.read().await;
        let record = match db.get_document(doc_id).await? {
            Some(r) => r,
            None => return Ok(None),
        };

        let content_store = self.content.lock().unwrap();
        let content = match content_store.get_document(doc_id)? {
            Some(c) => c,
            None => return Ok(None),
        };

        Ok(Some(Document {
            id: record.id,
            source_id: record.source_id,
            title: record.title,
            content,
            file_path: record.file_path,
            created_at: record.created_at,
            chunk_count: record.chunk_count,
        }))
    }

    /// Get all documents (for export)
    pub async fn get_all_documents(&self) -> anyhow::Result<Vec<Document>> {
        let db = self.db.read().await;
        let records = db.get_all_document_records(Some(db::MAX_QUERY_LIMIT)).await?;

        let content_store = self.content.lock().unwrap();
        let all_contents = content_store.get_all_documents()?;

        // Build a map of id -> content
        let content_map: HashMap<String, String> = all_contents
            .into_iter()
            .map(|(id, content, _)| (id, content))
            .collect();

        let documents: Vec<Document> = records
            .into_iter()
            .filter_map(|r| {
                let content = content_map.get(&r.id)?.clone();
                Some(Document {
                    id: r.id,
                    source_id: r.source_id,
                    title: r.title,
                    content,
                    file_path: r.file_path,
                    created_at: r.created_at,
                    chunk_count: r.chunk_count,
                })
            })
            .collect();

        Ok(documents)
    }

    /// Delete a document
    pub async fn delete_document(&self, doc_id: &str) -> anyhow::Result<()> {
        // Delete from LanceDB
        let db = self.db.read().await;
        db.delete_document(doc_id).await?;

        // Delete from SQLite
        let content = self.content.lock().unwrap();
        content.delete_document(doc_id)?;

        Ok(())
    }

    /// Delete a source
    pub async fn delete_source(&self, source_id: &str) -> anyhow::Result<()> {
        let db = self.db.read().await;

        // Get document IDs for this source (for SQLite cleanup)
        let doc_ids = db.get_document_ids_for_source(source_id).await?;

        // Delete from LanceDB
        db.delete_source(source_id).await?;

        // Delete from BM25 index
        self.bm25_index.delete_source(source_id)?;

        // Delete from SQLite
        let content = self.content.lock().unwrap();
        let doc_id_refs: Vec<&str> = doc_ids.iter().map(|s| s.as_str()).collect();
        content.delete_source(&doc_id_refs)?;

        Ok(())
    }

    /// Reset all data
    pub async fn reset_all(&self) -> anyhow::Result<()> {
        // Reset LanceDB
        let mut db = self.db.write().await;
        db.reset_all().await?;

        // Reset BM25 index
        self.bm25_index.reset()?;

        // Reset SQLite
        let content = self.content.lock().unwrap();
        content.reset()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // normalize_scores tests
    // ============================================================

    #[test]
    fn test_normalize_scores_basic() {
        let scores = vec![
            ("a".to_string(), 0.2),
            ("b".to_string(), 0.5),
            ("c".to_string(), 0.8),
        ];
        let normalized = Eywa::normalize_scores(&scores);

        assert_eq!(normalized.len(), 3);
        // Min (0.2) should map to 0.0
        assert!((normalized.iter().find(|(id, _)| id == "a").unwrap().1 - 0.0).abs() < 0.001);
        // Max (0.8) should map to 1.0
        assert!((normalized.iter().find(|(id, _)| id == "c").unwrap().1 - 1.0).abs() < 0.001);
        // Middle (0.5) should map to 0.5 (since range is 0.6, (0.5-0.2)/0.6 = 0.5)
        assert!((normalized.iter().find(|(id, _)| id == "b").unwrap().1 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_normalize_scores_empty() {
        let scores: Vec<(String, f32)> = vec![];
        let normalized = Eywa::normalize_scores(&scores);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_normalize_scores_single_item() {
        let scores = vec![("a".to_string(), 0.5)];
        let normalized = Eywa::normalize_scores(&scores);

        assert_eq!(normalized.len(), 1);
        // Single item has range 0, should return 1.0
        assert!((normalized[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_scores_all_same() {
        let scores = vec![
            ("a".to_string(), 0.5),
            ("b".to_string(), 0.5),
            ("c".to_string(), 0.5),
        ];
        let normalized = Eywa::normalize_scores(&scores);

        assert_eq!(normalized.len(), 3);
        // All same scores should all map to 1.0
        for (_, score) in &normalized {
            assert!((score - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_normalize_scores_preserves_order() {
        let scores = vec![
            ("first".to_string(), 0.1),
            ("second".to_string(), 0.9),
        ];
        let normalized = Eywa::normalize_scores(&scores);

        // Should preserve input order
        assert_eq!(normalized[0].0, "first");
        assert_eq!(normalized[1].0, "second");
    }

    #[test]
    fn test_normalize_scores_negative_values() {
        let scores = vec![
            ("a".to_string(), -1.0),
            ("b".to_string(), 0.0),
            ("c".to_string(), 1.0),
        ];
        let normalized = Eywa::normalize_scores(&scores);

        // Min (-1.0) -> 0.0, Max (1.0) -> 1.0
        assert!((normalized.iter().find(|(id, _)| id == "a").unwrap().1 - 0.0).abs() < 0.001);
        assert!((normalized.iter().find(|(id, _)| id == "c").unwrap().1 - 1.0).abs() < 0.001);
        assert!((normalized.iter().find(|(id, _)| id == "b").unwrap().1 - 0.5).abs() < 0.001);
    }

    // ============================================================
    // convex_fusion tests
    // ============================================================

    // Helper to create ChunkMeta for tests
    fn make_chunk_meta(id: &str, score: f32) -> ChunkMeta {
        ChunkMeta {
            id: id.to_string(),
            document_id: "doc1".to_string(),
            source_id: "src".to_string(),
            title: None,
            file_path: None,
            line_start: None,
            line_end: None,
            score,
        }
    }

    #[test]
    fn test_convex_fusion_weights_applied() {
        // Create mock ChunkMeta and BM25Result
        let vector_results = vec![make_chunk_meta("chunk1", 1.0)];
        let bm25_results = vec![
            BM25Result {
                chunk_id: "chunk1".to_string(),
                score: 1.0, // Will normalize to 1.0 (only item)
            },
        ];

        let fused = Eywa::convex_fusion(&vector_results, &bm25_results, 0.8, 0.2);

        assert_eq!(fused.len(), 1);
        // Both normalized to 1.0, so: 0.8 * 1.0 + 0.2 * 1.0 = 1.0
        assert!((fused[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_convex_fusion_overlapping_ids() {
        let vector_results = vec![
            make_chunk_meta("shared", 0.8),
            make_chunk_meta("vec_only", 0.4),
        ];
        let bm25_results = vec![
            BM25Result {
                chunk_id: "shared".to_string(),
                score: 0.9,
            },
            BM25Result {
                chunk_id: "bm25_only".to_string(),
                score: 0.5,
            },
        ];

        let fused = Eywa::convex_fusion(&vector_results, &bm25_results, 0.8, 0.2);

        assert_eq!(fused.len(), 3); // shared, vec_only, bm25_only

        // "shared" should have contributions from both
        let shared_score = fused.iter().find(|(id, _)| id == "shared").unwrap().1;
        // vec: normalized 0.8 -> 1.0 (max), bm25: normalized 0.9 -> 1.0 (max)
        // = 0.8 * 1.0 + 0.2 * 1.0 = 1.0
        assert!((shared_score - 1.0).abs() < 0.001);

        // "vec_only" should only have vector contribution
        let vec_only_score = fused.iter().find(|(id, _)| id == "vec_only").unwrap().1;
        // vec: normalized 0.4 -> 0.0 (min)
        // = 0.8 * 0.0 = 0.0
        assert!((vec_only_score - 0.0).abs() < 0.001);

        // "bm25_only" should only have BM25 contribution
        let bm25_only_score = fused.iter().find(|(id, _)| id == "bm25_only").unwrap().1;
        // bm25: normalized 0.5 -> 0.0 (min)
        // = 0.2 * 0.0 = 0.0
        assert!((bm25_only_score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_convex_fusion_disjoint_ids() {
        let vector_results = vec![make_chunk_meta("vec1", 0.9)];
        let bm25_results = vec![
            BM25Result {
                chunk_id: "bm25_1".to_string(),
                score: 0.7,
            },
        ];

        let fused = Eywa::convex_fusion(&vector_results, &bm25_results, 0.8, 0.2);

        assert_eq!(fused.len(), 2);
        // vec1: 0.8 * 1.0 = 0.8
        let vec1_score = fused.iter().find(|(id, _)| id == "vec1").unwrap().1;
        assert!((vec1_score - 0.8).abs() < 0.001);
        // bm25_1: 0.2 * 1.0 = 0.2
        let bm25_score = fused.iter().find(|(id, _)| id == "bm25_1").unwrap().1;
        assert!((bm25_score - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_convex_fusion_sorted_descending() {
        let vector_results = vec![
            make_chunk_meta("low", 0.2),
            make_chunk_meta("high", 0.9),
        ];
        let bm25_results: Vec<BM25Result> = vec![];

        let fused = Eywa::convex_fusion(&vector_results, &bm25_results, 0.8, 0.2);

        // Results should be sorted by score descending
        assert_eq!(fused[0].0, "high");
        assert_eq!(fused[1].0, "low");
    }

    #[test]
    fn test_convex_fusion_empty_vector() {
        let vector_results: Vec<ChunkMeta> = vec![];
        let bm25_results = vec![
            BM25Result {
                chunk_id: "bm25".to_string(),
                score: 0.8,
            },
        ];

        let fused = Eywa::convex_fusion(&vector_results, &bm25_results, 0.8, 0.2);

        assert_eq!(fused.len(), 1);
        // Only BM25 contribution: 0.2 * 1.0 = 0.2
        assert!((fused[0].1 - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_convex_fusion_empty_bm25() {
        let vector_results = vec![make_chunk_meta("vec", 0.8)];
        let bm25_results: Vec<BM25Result> = vec![];

        let fused = Eywa::convex_fusion(&vector_results, &bm25_results, 0.8, 0.2);

        assert_eq!(fused.len(), 1);
        // Only vector contribution: 0.8 * 1.0 = 0.8
        assert!((fused[0].1 - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_convex_fusion_both_empty() {
        let vector_results: Vec<ChunkMeta> = vec![];
        let bm25_results: Vec<BM25Result> = vec![];

        let fused = Eywa::convex_fusion(&vector_results, &bm25_results, 0.8, 0.2);

        assert!(fused.is_empty());
    }
}

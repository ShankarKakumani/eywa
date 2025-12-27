//! Batch Accumulator for document and chunk collection
//!
//! Holds documents and chunks in memory until batch thresholds are reached.
//! This prevents LanceDB fragmentation by ensuring large batch writes.

use super::{BatchConfig, ChunkData, PreparedDoc};

/// Accumulates documents and chunks until batch thresholds are reached
pub struct BatchAccumulator {
    /// Accumulated documents
    documents: Vec<PreparedDoc>,
    /// Total chunks across all documents
    total_chunks: usize,
    /// Estimated memory usage in bytes
    memory_bytes: usize,
    /// Configuration thresholds
    config: BatchConfig,
}

impl BatchAccumulator {
    /// Create a new accumulator with the given config
    pub fn new(config: BatchConfig) -> Self {
        Self {
            documents: Vec::new(),
            total_chunks: 0,
            memory_bytes: 0,
            config,
        }
    }

    /// Add a document to the accumulator
    ///
    /// Returns true if the batch should be flushed after this addition
    pub fn add_document(&mut self, doc: PreparedDoc) -> bool {
        // Estimate memory for this document
        let doc_memory = Self::estimate_doc_memory(&doc);

        self.total_chunks += doc.chunks.len();
        self.memory_bytes += doc_memory;
        self.documents.push(doc);

        self.should_flush()
    }

    /// Check if the batch should be flushed based on thresholds
    pub fn should_flush(&self) -> bool {
        // Flush if any threshold is exceeded
        self.documents.len() >= self.config.max_docs
            || self.total_chunks >= self.config.max_chunks
            || self.memory_bytes >= self.config.max_memory_mb * 1024 * 1024
    }

    /// Get the number of accumulated documents
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Get the total number of chunks
    pub fn chunk_count(&self) -> usize {
        self.total_chunks
    }

    /// Get estimated memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.memory_bytes
    }

    /// Get all chunks from accumulated documents
    pub fn all_chunks(&self) -> Vec<&ChunkData> {
        self.documents
            .iter()
            .flat_map(|doc| doc.chunks.iter())
            .collect()
    }

    /// Take all documents out of the accumulator, resetting it
    pub fn take_documents(&mut self) -> Vec<PreparedDoc> {
        self.total_chunks = 0;
        self.memory_bytes = 0;
        std::mem::take(&mut self.documents)
    }

    /// Check if the accumulator is empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Estimate memory usage for a document
    fn estimate_doc_memory(doc: &PreparedDoc) -> usize {
        // Document content + metadata
        let doc_size = doc.content.len()
            + doc.title.len()
            + doc.file_path.as_ref().map_or(0, |p| p.len())
            + doc.id.len()
            + doc.created_at.len()
            + 32; // overhead for struct fields

        // Chunk content + metadata
        let chunks_size: usize = doc.chunks.iter().map(|c| {
            c.content.len()
                + c.id.len()
                + c.document_id.len()
                + c.source_id.len()
                + c.title.as_ref().map_or(0, |t| t.len())
                + c.file_path.as_ref().map_or(0, |p| p.len())
                + c.content_hash.len()
                + 48 // overhead for struct fields
        }).sum();

        doc_size + chunks_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_doc(num_chunks: usize) -> PreparedDoc {
        let chunks: Vec<ChunkData> = (0..num_chunks)
            .map(|i| ChunkData {
                id: format!("chunk-{}", i),
                document_id: "doc-1".to_string(),
                source_id: "test".to_string(),
                title: Some("Test".to_string()),
                content: "x".repeat(100),
                file_path: None,
                line_start: 1,
                line_end: 10,
                content_hash: format!("hash-{}", i),
                section: None,
                subsection: None,
                hierarchy: Vec::new(),
                has_code: false,
            })
            .collect();

        PreparedDoc {
            id: "doc-1".to_string(),
            content: "Test content".to_string(),
            title: "Test".to_string(),
            file_path: None,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            content_length: 12,
            chunks,
        }
    }

    #[test]
    fn test_accumulator_basic() {
        let config = BatchConfig {
            max_docs: 10,
            max_chunks: 100,
            max_memory_mb: 10,
            flush_timeout_secs: 5,
        };
        let mut acc = BatchAccumulator::new(config);

        assert!(acc.is_empty());
        assert_eq!(acc.document_count(), 0);
        assert_eq!(acc.chunk_count(), 0);

        let doc = make_test_doc(5);
        let should_flush = acc.add_document(doc);

        assert!(!should_flush);
        assert!(!acc.is_empty());
        assert_eq!(acc.document_count(), 1);
        assert_eq!(acc.chunk_count(), 5);
    }

    #[test]
    fn test_accumulator_flush_on_docs() {
        let config = BatchConfig {
            max_docs: 2,
            max_chunks: 1000,
            max_memory_mb: 100,
            flush_timeout_secs: 5,
        };
        let mut acc = BatchAccumulator::new(config);

        acc.add_document(make_test_doc(1));
        let should_flush = acc.add_document(make_test_doc(1));

        assert!(should_flush);
    }

    #[test]
    fn test_accumulator_flush_on_chunks() {
        let config = BatchConfig {
            max_docs: 100,
            max_chunks: 10,
            max_memory_mb: 100,
            flush_timeout_secs: 5,
        };
        let mut acc = BatchAccumulator::new(config);

        let should_flush = acc.add_document(make_test_doc(15));

        assert!(should_flush);
    }

    #[test]
    fn test_take_documents() {
        let config = BatchConfig::default();
        let mut acc = BatchAccumulator::new(config);

        acc.add_document(make_test_doc(5));
        acc.add_document(make_test_doc(3));

        assert_eq!(acc.document_count(), 2);
        assert_eq!(acc.chunk_count(), 8);

        let docs = acc.take_documents();

        assert_eq!(docs.len(), 2);
        assert!(acc.is_empty());
        assert_eq!(acc.chunk_count(), 0);
    }
}

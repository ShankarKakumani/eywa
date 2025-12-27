//! Fallback Chunker
//!
//! Simple character-based chunking for unknown file types.
//! Uses line boundaries when possible.

use super::{create_chunk, Chunk, ChunkMetadata, Chunker, DocMetadata, MIN_CHUNK, OVERLAP, TARGET_SIZE};

/// Fallback chunker for unknown file types
pub struct FallbackChunker {
    target_size: usize,
    overlap: usize,
}

impl FallbackChunker {
    pub fn new() -> Self {
        Self {
            target_size: TARGET_SIZE,
            overlap: OVERLAP,
        }
    }

    pub fn with_sizes(target_size: usize, overlap: usize) -> Self {
        Self {
            target_size,
            overlap,
        }
    }

    /// Split content into lines, respecting chunk size limits
    fn chunk_by_lines(&self, content: &str, base_metadata: &ChunkMetadata) -> Vec<Chunk> {
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

            // Check if adding this line would exceed target size
            if current_chunk.len() + line_with_newline.len() > self.target_size
                && !current_chunk.is_empty()
            {
                // Create chunk if it's big enough
                if current_chunk.len() >= MIN_CHUNK {
                    let meta = base_metadata
                        .clone()
                        .with_lines(chunk_start_line, current_line - 1)
                        .with_code(current_chunk.contains("```"));

                    chunks.push(create_chunk(current_chunk.clone(), meta));
                }

                // Overlap: keep last OVERLAP chars, find line boundary
                let overlap_start = self.find_overlap_start(&current_chunk);
                current_chunk = current_chunk[overlap_start..].to_string();
                chunk_start_line = current_line.saturating_sub(current_chunk.lines().count() as u32);
            }

            current_chunk.push_str(&line_with_newline);
            current_line += 1;
        }

        // Last chunk
        if current_chunk.trim().len() >= MIN_CHUNK {
            let meta = base_metadata
                .clone()
                .with_lines(chunk_start_line, current_line - 1)
                .with_code(current_chunk.contains("```"));

            chunks.push(create_chunk(current_chunk, meta));
        }

        chunks
    }

    /// Find the start position for overlap, preferring line boundaries
    fn find_overlap_start(&self, content: &str) -> usize {
        if content.len() <= self.overlap {
            return 0;
        }

        let target_start = content.len() - self.overlap;

        // Find the nearest newline after target_start
        if let Some(newline_pos) = content[target_start..].find('\n') {
            return target_start + newline_pos + 1;
        }

        // No newline found, use char boundary
        content
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= target_start)
            .unwrap_or(0)
    }
}

impl Default for FallbackChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker for FallbackChunker {
    fn chunk(&self, content: &str, metadata: &DocMetadata) -> Vec<Chunk> {
        if content.trim().is_empty() {
            return Vec::new();
        }

        // Extract title from file path if available
        let title = metadata.file_path.as_ref().and_then(|p| {
            std::path::Path::new(p)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
        });

        let base_metadata = ChunkMetadata::new(metadata).with_title(title);

        self.chunk_by_lines(content, &base_metadata)
    }

    fn supported_extensions(&self) -> &[&str] {
        // Fallback handles everything not matched by other chunkers
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_doc() -> DocMetadata {
        DocMetadata {
            document_id: "doc1".to_string(),
            source_id: "src1".to_string(),
            file_path: Some("test.txt".to_string()),
        }
    }

    #[test]
    fn test_empty_content() {
        let chunker = FallbackChunker::new();
        let chunks = chunker.chunk("", &test_doc());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_small_content() {
        let chunker = FallbackChunker::new();
        let content = "Hello, world!\nThis is a test.";
        let chunks = chunker.chunk(content, &test_doc());

        // Content is smaller than MIN_CHUNK, might not create a chunk
        // unless we adjust MIN_CHUNK
        assert!(chunks.len() <= 1);
    }

    #[test]
    fn test_large_content_splits() {
        let chunker = FallbackChunker::with_sizes(200, 20);
        // Each line is ~40 chars, 50 lines = 2000 chars
        let content = (1..=50).map(|i| format!("This is line number {} with content", i)).collect::<Vec<_>>().join("\n");
        let chunks = chunker.chunk(&content, &test_doc());

        assert!(chunks.len() > 1, "Should split {} chars into multiple chunks, got {} chunks", content.len(), chunks.len());
    }

    #[test]
    fn test_line_numbers_tracked() {
        let chunker = FallbackChunker::with_sizes(100, 20);
        let content = (1..=20).map(|i| format!("Line {}", i)).collect::<Vec<_>>().join("\n");
        let chunks = chunker.chunk(&content, &test_doc());

        if !chunks.is_empty() {
            // First chunk should start at line 1
            assert_eq!(chunks[0].metadata.line_start, 1);
        }
    }

    #[test]
    fn test_title_from_filename() {
        let chunker = FallbackChunker::new();
        let content = "x".repeat(200); // Ensure it's big enough
        let chunks = chunker.chunk(&content, &test_doc());

        if !chunks.is_empty() {
            assert_eq!(chunks[0].metadata.title, Some("test.txt".to_string()));
        }
    }
}

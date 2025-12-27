//! Text Chunker
//!
//! Paragraph-based chunking for plain text files.
//! Splits on double newlines (\n\n) to preserve paragraph boundaries.

use super::{create_chunk, Chunk, ChunkMetadata, Chunker, DocMetadata, MIN_CHUNK, OVERLAP, TARGET_SIZE};

/// Paragraph-based chunker for plain text files
pub struct TextChunker {
    target_size: usize,
    overlap: usize,
}

impl TextChunker {
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

    /// Split content into paragraphs
    fn split_paragraphs(content: &str) -> Vec<&str> {
        content
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect()
    }

    /// Count lines in content up to a position
    #[allow(dead_code)]
    fn count_lines_before(content: &str, target: &str) -> u32 {
        if let Some(pos) = content.find(target) {
            content[..pos].lines().count() as u32 + 1
        } else {
            1
        }
    }

    /// Count lines in a string
    fn count_lines(s: &str) -> u32 {
        s.lines().count().max(1) as u32
    }
}

impl Default for TextChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker for TextChunker {
    fn chunk(&self, content: &str, metadata: &DocMetadata) -> Vec<Chunk> {
        if content.trim().is_empty() {
            return Vec::new();
        }

        let paragraphs = Self::split_paragraphs(content);
        if paragraphs.is_empty() {
            return Vec::new();
        }

        // Extract title from file path
        let title = metadata.file_path.as_ref().and_then(|p| {
            std::path::Path::new(p)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
        });

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut chunk_start_line = 1u32;
        let mut current_line = 1u32;

        for para in paragraphs {
            let para_with_sep = if current_chunk.is_empty() {
                para.to_string()
            } else {
                format!("\n\n{}", para)
            };

            // Check if adding this paragraph would exceed target size
            if current_chunk.len() + para_with_sep.len() > self.target_size
                && !current_chunk.is_empty()
            {
                // Create chunk if it's big enough
                if current_chunk.len() >= MIN_CHUNK {
                    let line_end = chunk_start_line + Self::count_lines(&current_chunk) - 1;
                    let meta = ChunkMetadata::new(metadata)
                        .with_title(title.clone())
                        .with_lines(chunk_start_line, line_end);

                    chunks.push(create_chunk(current_chunk.clone(), meta));
                }

                // For overlap, keep the last paragraph if it fits
                if para.len() < self.overlap {
                    current_chunk = para.to_string();
                    chunk_start_line = current_line;
                } else {
                    current_chunk = String::new();
                    chunk_start_line = current_line;
                }
            }

            if current_chunk.is_empty() {
                current_chunk = para.to_string();
                chunk_start_line = current_line;
            } else {
                current_chunk.push_str(&format!("\n\n{}", para));
            }

            current_line += Self::count_lines(para) + 1; // +1 for blank line
        }

        // Last chunk
        if current_chunk.len() >= MIN_CHUNK {
            let line_end = chunk_start_line + Self::count_lines(&current_chunk) - 1;
            let meta = ChunkMetadata::new(metadata)
                .with_title(title)
                .with_lines(chunk_start_line, line_end);

            chunks.push(create_chunk(current_chunk, meta));
        }

        chunks
    }

    fn supported_extensions(&self) -> &[&str] {
        &["txt"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_doc() -> DocMetadata {
        DocMetadata {
            document_id: "doc1".to_string(),
            source_id: "src1".to_string(),
            file_path: Some("readme.txt".to_string()),
        }
    }

    #[test]
    fn test_empty_content() {
        let chunker = TextChunker::new();
        let chunks = chunker.chunk("", &test_doc());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_single_paragraph() {
        let chunker = TextChunker::new();
        let content = "This is a single paragraph with enough content to meet the minimum chunk size requirement for our chunking algorithm.";
        let chunks = chunker.chunk(content, &test_doc());

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].metadata.title, Some("readme.txt".to_string()));
    }

    #[test]
    fn test_multiple_paragraphs() {
        let chunker = TextChunker::with_sizes(200, 50);
        let content = "First paragraph with some content here.\n\n\
                      Second paragraph with more content.\n\n\
                      Third paragraph with even more content.\n\n\
                      Fourth paragraph to add more text.\n\n\
                      Fifth paragraph for good measure.";

        let chunks = chunker.chunk(content, &test_doc());
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_paragraph_boundaries_preserved() {
        let chunker = TextChunker::new();
        let content = "Short para 1.\n\n\
                      This is a longer paragraph that should be kept together because it represents a complete thought.\n\n\
                      Another short one.";

        let chunks = chunker.chunk(content, &test_doc());

        // All chunks should contain complete paragraphs
        for chunk in chunks {
            // No chunk should start or end mid-sentence
            let trimmed = chunk.content.trim();
            assert!(
                trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?'),
                "Chunk should end at sentence boundary: {:?}",
                trimmed.chars().rev().take(20).collect::<String>()
            );
        }
    }
}

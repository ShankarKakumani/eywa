//! PDF Chunker
//!
//! Stub implementation for PDF files.
//! Currently delegates to MarkdownChunker for text extraction.
//! Future: Use pdf_oxide for proper PDF → Markdown conversion.

use super::{Chunk, Chunker, DocMetadata, MarkdownChunker};

/// PDF chunker (stub - delegates to markdown chunker)
pub struct PdfChunker {
    md_chunker: MarkdownChunker,
}

impl PdfChunker {
    pub fn new() -> Self {
        Self {
            md_chunker: MarkdownChunker::new(),
        }
    }
}

impl Default for PdfChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker for PdfChunker {
    fn chunk(&self, content: &str, metadata: &DocMetadata) -> Vec<Chunk> {
        // For now, treat extracted PDF text as markdown
        // This works because:
        // - PDF text extraction often preserves some structure
        // - MarkdownChunker handles plain text gracefully
        //
        // Future enhancement: Use pdf_oxide to convert PDF to Markdown
        // which preserves headings, tables, and structure better
        self.md_chunker.chunk(content, metadata)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_doc() -> DocMetadata {
        DocMetadata {
            document_id: "doc1".to_string(),
            source_id: "src1".to_string(),
            file_path: Some("document.pdf".to_string()),
        }
    }

    #[test]
    fn test_pdf_chunker_delegates_to_markdown() {
        let chunker = PdfChunker::new();
        let content = "# Document Title\n\nSome extracted PDF content here.";

        let chunks = chunker.chunk(content, &test_doc());

        // Should produce chunks like markdown chunker would
        assert!(!chunks.is_empty() || content.len() < super::super::MIN_CHUNK);
    }

    #[test]
    fn test_supported_extensions() {
        let chunker = PdfChunker::new();
        assert_eq!(chunker.supported_extensions(), &["pdf"]);
    }
}

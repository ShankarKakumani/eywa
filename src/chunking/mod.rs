//! Smart Chunking for Eywa
//!
//! Content-aware chunking strategies for different file types:
//! - Markdown: Header-aware with hierarchical metadata
//! - Text: Paragraph-based splitting
//! - PDF: Text extraction (stub for now)
//! - Fallback: Recursive char-based for unknown types

pub mod fallback;
pub mod markdown;
pub mod pdf;
pub mod text;

pub use fallback::FallbackChunker;
pub use markdown::MarkdownChunker;
pub use pdf::PdfChunker;
pub use text::TextChunker;

use std::path::Path;

/// Chunk size parameters
pub const TARGET_SIZE: usize = 1500; // ~400-512 tokens
pub const OVERLAP: usize = 200; // ~10-15%
pub const MIN_CHUNK: usize = 100; // Skip tiny chunks
pub const MAX_CHUNK: usize = 3000; // Hard limit

/// Document metadata for chunking context
#[derive(Debug, Clone)]
pub struct DocMetadata {
    pub document_id: String,
    pub source_id: String,
    pub file_path: Option<String>,
}

/// A chunk of content with metadata
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub content: String,
    pub metadata: ChunkMetadata,
}

/// Metadata for a chunk
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub document_id: String,
    pub source_id: String,
    pub file_path: Option<String>,
    /// H1 or filename
    pub title: Option<String>,
    /// H2
    pub section: Option<String>,
    /// H3
    pub subsection: Option<String>,
    /// Full hierarchy path: ["Title", "Section", "Subsection"]
    pub hierarchy: Vec<String>,
    /// Contains code blocks
    pub has_code: bool,
    pub line_start: u32,
    pub line_end: u32,
    pub content_hash: String,
}

impl ChunkMetadata {
    /// Create new metadata with defaults
    pub fn new(doc: &DocMetadata) -> Self {
        Self {
            document_id: doc.document_id.clone(),
            source_id: doc.source_id.clone(),
            file_path: doc.file_path.clone(),
            title: None,
            section: None,
            subsection: None,
            hierarchy: Vec::new(),
            has_code: false,
            line_start: 1,
            line_end: 1,
            content_hash: String::new(),
        }
    }

    /// Set title and update hierarchy
    pub fn with_title(mut self, title: Option<String>) -> Self {
        self.title = title.clone();
        if let Some(t) = title {
            if self.hierarchy.is_empty() || self.hierarchy[0] != t {
                self.hierarchy.insert(0, t);
            }
        }
        self
    }

    /// Set section and update hierarchy
    pub fn with_section(mut self, section: Option<String>) -> Self {
        self.section = section.clone();
        if let Some(s) = section {
            // Ensure we have title first, then section
            while self.hierarchy.len() < 1 {
                self.hierarchy.push(String::new());
            }
            if self.hierarchy.len() == 1 {
                self.hierarchy.push(s);
            } else {
                self.hierarchy[1] = s;
            }
        }
        self
    }

    /// Set subsection and update hierarchy
    pub fn with_subsection(mut self, subsection: Option<String>) -> Self {
        self.subsection = subsection.clone();
        if let Some(s) = subsection {
            while self.hierarchy.len() < 2 {
                self.hierarchy.push(String::new());
            }
            if self.hierarchy.len() == 2 {
                self.hierarchy.push(s);
            } else {
                self.hierarchy[2] = s;
            }
        }
        self
    }

    /// Set line range
    pub fn with_lines(mut self, start: u32, end: u32) -> Self {
        self.line_start = start;
        self.line_end = end;
        self
    }

    /// Set has_code flag
    pub fn with_code(mut self, has_code: bool) -> Self {
        self.has_code = has_code;
        self
    }

    /// Compute and set content hash
    pub fn with_hash(mut self, content: &str) -> Self {
        self.content_hash = format!("{:x}", md5::compute(content.as_bytes()));
        self
    }
}

/// Trait for content-aware chunking
pub trait Chunker: Send + Sync {
    /// Chunk content into pieces with metadata
    fn chunk(&self, content: &str, metadata: &DocMetadata) -> Vec<Chunk>;

    /// File extensions this chunker handles
    fn supported_extensions(&self) -> &[&str];
}

/// Registry of chunkers, picks the right one based on file extension
pub struct ChunkerRegistry {
    markdown: MarkdownChunker,
    text: TextChunker,
    pdf: PdfChunker,
    fallback: FallbackChunker,
}

impl ChunkerRegistry {
    /// Create a new registry with default chunkers
    pub fn new() -> Self {
        Self {
            markdown: MarkdownChunker::new(),
            text: TextChunker::new(),
            pdf: PdfChunker::new(),
            fallback: FallbackChunker::new(),
        }
    }

    /// Get file extension from path
    fn get_extension(file_path: &str) -> Option<String> {
        Path::new(file_path)
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
    }

    /// Chunk content using the appropriate chunker
    pub fn chunk(&self, content: &str, file_path: Option<&str>, metadata: &DocMetadata) -> Vec<Chunk> {
        let ext = file_path
            .and_then(Self::get_extension)
            .unwrap_or_default();

        match ext.as_str() {
            "md" | "markdown" => self.markdown.chunk(content, metadata),
            "txt" => self.text.chunk(content, metadata),
            "pdf" => self.pdf.chunk(content, metadata),
            _ => self.fallback.chunk(content, metadata),
        }
    }
}

impl Default for ChunkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create a chunk with a new UUID
pub fn create_chunk(content: String, metadata: ChunkMetadata) -> Chunk {
    let mut meta = metadata;
    meta.content_hash = format!("{:x}", md5::compute(content.as_bytes()));

    Chunk {
        id: uuid::Uuid::new_v4().to_string(),
        content,
        metadata: meta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_metadata_hierarchy() {
        let doc = DocMetadata {
            document_id: "doc1".to_string(),
            source_id: "src1".to_string(),
            file_path: Some("test.md".to_string()),
        };

        let meta = ChunkMetadata::new(&doc)
            .with_title(Some("My Doc".to_string()))
            .with_section(Some("Getting Started".to_string()))
            .with_subsection(Some("Installation".to_string()));

        assert_eq!(meta.title, Some("My Doc".to_string()));
        assert_eq!(meta.section, Some("Getting Started".to_string()));
        assert_eq!(meta.subsection, Some("Installation".to_string()));
        assert_eq!(meta.hierarchy, vec!["My Doc", "Getting Started", "Installation"]);
    }

    #[test]
    fn test_registry_extension_matching() {
        let registry = ChunkerRegistry::new();
        let doc = DocMetadata {
            document_id: "doc1".to_string(),
            source_id: "src1".to_string(),
            file_path: Some("test.md".to_string()),
        };

        // Create content large enough to exceed MIN_CHUNK (100 chars)
        let content = format!("# Hello\n\n{}", "This is test content. ".repeat(10));
        let chunks = registry.chunk(&content, Some("test.md"), &doc);
        assert!(!chunks.is_empty());
    }
}

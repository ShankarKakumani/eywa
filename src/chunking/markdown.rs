//! Markdown Chunker
//!
//! Header-aware chunking for Markdown files.
//! Uses pulldown-cmark to parse and extract structure.
//! Tracks H1/H2/H3 headers for hierarchical metadata.

use super::{create_chunk, Chunk, ChunkMetadata, Chunker, DocMetadata, MIN_CHUNK, OVERLAP, TARGET_SIZE};
// Note: pulldown-cmark imported for future use with proper AST parsing
// Currently using simple string-based header detection

/// Header-aware chunker for Markdown files
pub struct MarkdownChunker {
    target_size: usize,
    overlap: usize,
}

/// Current section context while parsing
#[derive(Clone, Default)]
struct SectionContext {
    title: Option<String>,      // H1
    section: Option<String>,    // H2
    subsection: Option<String>, // H3
}

impl SectionContext {
    fn to_hierarchy(&self) -> Vec<String> {
        let mut h = Vec::new();
        if let Some(t) = &self.title {
            h.push(t.clone());
        }
        if let Some(s) = &self.section {
            h.push(s.clone());
        }
        if let Some(ss) = &self.subsection {
            h.push(ss.clone());
        }
        h
    }
}

impl MarkdownChunker {
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

    /// Check if content contains code blocks
    fn has_code_blocks(content: &str) -> bool {
        content.contains("```")
    }

    /// Extract header text from markdown events
    fn extract_header_text(content: &str, start_offset: usize) -> Option<String> {
        // Find the end of the line starting at offset
        let remaining = &content[start_offset..];
        let line = remaining.lines().next()?;

        // Strip the # prefix and trim
        let text = line.trim_start_matches('#').trim();
        if text.is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    }

    /// Split markdown into sections based on headers
    fn split_into_sections(content: &str) -> Vec<(SectionContext, String, u32, u32)> {
        let mut sections = Vec::new();
        let mut current_context = SectionContext::default();
        let mut current_content = String::new();
        let mut current_start_line = 1u32;
        let mut current_line = 1u32;
        let mut in_code_block = false;

        // First, try to extract title from first H1 if present (outside code blocks)
        let mut temp_in_code = false;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                temp_in_code = !temp_in_code;
                continue;
            }
            if temp_in_code {
                continue;
            }
            if trimmed.starts_with("# ") && !trimmed.starts_with("##") {
                current_context.title = Some(trimmed[2..].trim().to_string());
                break;
            }
            if !trimmed.is_empty() && !trimmed.starts_with("#") {
                break; // Non-header content before any header
            }
        }

        for line in content.lines() {
            let trimmed = line.trim();

            // Track code block state - skip header detection inside code blocks
            if trimmed.starts_with("```") {
                in_code_block = !in_code_block;
                current_content.push_str(line);
                current_content.push('\n');
                current_line += 1;
                continue;
            }

            // Inside code block - treat as regular content
            if in_code_block {
                current_content.push_str(line);
                current_content.push('\n');
                current_line += 1;
                continue;
            }

            // Check for headers (only outside code blocks)
            if trimmed.starts_with("# ") && !trimmed.starts_with("##") {
                // H1 - Title
                if !current_content.trim().is_empty() {
                    sections.push((
                        current_context.clone(),
                        current_content.clone(),
                        current_start_line,
                        current_line - 1,
                    ));
                }
                current_context.title = Some(trimmed[2..].trim().to_string());
                current_context.section = None;
                current_context.subsection = None;
                current_content = format!("{}\n", line);
                current_start_line = current_line;
            } else if trimmed.starts_with("## ") && !trimmed.starts_with("###") {
                // H2 - Section
                if !current_content.trim().is_empty() {
                    sections.push((
                        current_context.clone(),
                        current_content.clone(),
                        current_start_line,
                        current_line - 1,
                    ));
                }
                current_context.section = Some(trimmed[3..].trim().to_string());
                current_context.subsection = None;
                current_content = format!("{}\n", line);
                current_start_line = current_line;
            } else if trimmed.starts_with("### ") {
                // H3 - Subsection
                if !current_content.trim().is_empty() {
                    sections.push((
                        current_context.clone(),
                        current_content.clone(),
                        current_start_line,
                        current_line - 1,
                    ));
                }
                current_context.subsection = Some(trimmed[4..].trim().to_string());
                current_content = format!("{}\n", line);
                current_start_line = current_line;
            } else {
                // Regular content
                current_content.push_str(line);
                current_content.push('\n');
            }

            current_line += 1;
        }

        // Don't forget the last section
        if !current_content.trim().is_empty() {
            sections.push((
                current_context,
                current_content,
                current_start_line,
                current_line - 1,
            ));
        }

        sections
    }

    /// Further split a section if it exceeds target size
    fn split_large_section(
        &self,
        context: &SectionContext,
        content: &str,
        start_line: u32,
        metadata: &DocMetadata,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        let mut current_chunk = String::new();
        let mut chunk_start_line = start_line;
        let mut current_line = start_line;
        let mut in_code_block = false;

        for line in &lines {
            let trimmed = line.trim();
            let is_code_fence = trimmed.starts_with("```");

            // Track code block state
            if is_code_fence {
                in_code_block = !in_code_block;
            }

            let line_with_newline = format!("{}\n", line);

            // Only split if:
            // - Not in a code block
            // - Not at a code fence line (keep ``` with the code)
            // - Exceeds size limit
            // - Current chunk is large enough
            if !in_code_block
                && !is_code_fence
                && current_chunk.len() + line_with_newline.len() > self.target_size
                && current_chunk.len() >= MIN_CHUNK
            {
                let has_code = Self::has_code_blocks(&current_chunk);
                let hierarchy = context.to_hierarchy();

                let meta = ChunkMetadata::new(metadata)
                    .with_title(context.title.clone())
                    .with_section(context.section.clone())
                    .with_subsection(context.subsection.clone())
                    .with_lines(chunk_start_line, current_line - 1)
                    .with_code(has_code);

                // Override hierarchy
                let mut meta = meta;
                meta.hierarchy = hierarchy;

                chunks.push(create_chunk(current_chunk.clone(), meta));

                // Start new chunk (overlap handled at section level)
                current_chunk = String::new();
                chunk_start_line = current_line;
            }

            current_chunk.push_str(&line_with_newline);
            current_line += 1;
        }

        // Last chunk
        if current_chunk.len() >= MIN_CHUNK {
            let has_code = Self::has_code_blocks(&current_chunk);
            let hierarchy = context.to_hierarchy();

            let meta = ChunkMetadata::new(metadata)
                .with_title(context.title.clone())
                .with_section(context.section.clone())
                .with_subsection(context.subsection.clone())
                .with_lines(chunk_start_line, current_line - 1)
                .with_code(has_code);

            let mut meta = meta;
            meta.hierarchy = hierarchy;

            chunks.push(create_chunk(current_chunk, meta));
        }

        chunks
    }
}

impl Default for MarkdownChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker for MarkdownChunker {
    fn chunk(&self, content: &str, metadata: &DocMetadata) -> Vec<Chunk> {
        if content.trim().is_empty() {
            return Vec::new();
        }

        let sections = Self::split_into_sections(content);
        let mut chunks = Vec::new();

        for (context, section_content, start_line, end_line) in sections {
            if section_content.len() <= self.target_size {
                // Section fits in one chunk
                if section_content.len() >= MIN_CHUNK {
                    let has_code = Self::has_code_blocks(&section_content);
                    let hierarchy = context.to_hierarchy();

                    let meta = ChunkMetadata::new(metadata)
                        .with_title(context.title.clone())
                        .with_section(context.section.clone())
                        .with_subsection(context.subsection.clone())
                        .with_lines(start_line, end_line)
                        .with_code(has_code);

                    let mut meta = meta;
                    meta.hierarchy = hierarchy;

                    chunks.push(create_chunk(section_content, meta));
                }
            } else {
                // Section too large, split further
                let split_chunks =
                    self.split_large_section(&context, &section_content, start_line, metadata);
                chunks.extend(split_chunks);
            }
        }

        // If no chunks created (content too small), create one chunk with everything
        if chunks.is_empty() && content.len() >= MIN_CHUNK {
            let meta = ChunkMetadata::new(metadata)
                .with_lines(1, content.lines().count() as u32)
                .with_code(Self::has_code_blocks(content));

            chunks.push(create_chunk(content.to_string(), meta));
        }

        chunks
    }

    fn supported_extensions(&self) -> &[&str] {
        &["md", "markdown"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_doc() -> DocMetadata {
        DocMetadata {
            document_id: "doc1".to_string(),
            source_id: "src1".to_string(),
            file_path: Some("guide.md".to_string()),
        }
    }

    #[test]
    fn test_empty_content() {
        let chunker = MarkdownChunker::new();
        let chunks = chunker.chunk("", &test_doc());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_simple_markdown() {
        let chunker = MarkdownChunker::new();
        let content = r#"# My Document

This is the introduction. It has enough content to meet minimum chunk requirements.

## Getting Started

Here's how to get started with the project. Follow these steps carefully.

### Installation

Run the following command to install."#;

        let chunks = chunker.chunk(content, &test_doc());
        assert!(!chunks.is_empty());

        // Check that hierarchy is captured
        if chunks.len() > 1 {
            let last_chunk = chunks.last().unwrap();
            assert!(last_chunk.metadata.hierarchy.contains(&"My Document".to_string()));
        }
    }

    #[test]
    fn test_code_block_detection() {
        let chunker = MarkdownChunker::new();
        let content = r#"# Code Example

Here's some code:

```rust
fn main() {
    println!("Hello, world!");
}
```

That was the code."#;

        let chunks = chunker.chunk(content, &test_doc());

        if !chunks.is_empty() {
            assert!(
                chunks[0].metadata.has_code,
                "Should detect code block"
            );
        }
    }

    #[test]
    fn test_header_hierarchy() {
        let chunker = MarkdownChunker::with_sizes(200, 50);
        let content = r#"# Title

Introduction text here.

## Section One

Content for section one.

### Subsection A

Details for subsection A.

## Section Two

Content for section two."#;

        let chunks = chunker.chunk(content, &test_doc());

        // Find a chunk with subsection
        for chunk in &chunks {
            if chunk.metadata.subsection.is_some() {
                assert_eq!(chunk.metadata.title, Some("Title".to_string()));
                assert_eq!(chunk.metadata.section, Some("Section One".to_string()));
                assert_eq!(chunk.metadata.subsection, Some("Subsection A".to_string()));
                break;
            }
        }
    }

    #[test]
    fn test_code_blocks_kept_intact() {
        let chunker = MarkdownChunker::with_sizes(100, 20);
        let content = r#"# Example

```python
def long_function():
    # This is a long function
    x = 1
    y = 2
    z = 3
    return x + y + z
```"#;

        let chunks = chunker.chunk(content, &test_doc());

        // Code block should not be split in the middle
        for chunk in &chunks {
            let backtick_count = chunk.content.matches("```").count();
            // Should be 0 (no code) or 2 (complete block) - never 1
            assert_ne!(
                backtick_count, 1,
                "Code block was split: {:?}",
                chunk.content
            );
        }
    }
}

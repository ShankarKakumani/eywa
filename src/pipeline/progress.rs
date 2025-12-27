//! Progress Tracker for ingestion pipeline
//!
//! Provides real-time progress updates during document ingestion.

use std::io::{self, Write};

/// Tracks and displays progress during ingestion
pub struct ProgressTracker {
    /// Total number of documents to process
    total_docs: usize,
    /// Number of documents processed
    processed_docs: usize,
    /// Number of chunks processed
    processed_chunks: usize,
    /// Current phase name
    current_phase: Option<String>,
    /// Whether to show output (false for tests/quiet mode)
    show_output: bool,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(total_docs: usize) -> Self {
        Self {
            total_docs,
            processed_docs: 0,
            processed_chunks: 0,
            current_phase: None,
            show_output: true,
        }
    }

    /// Create a quiet progress tracker (no output)
    pub fn quiet(total_docs: usize) -> Self {
        Self {
            total_docs,
            processed_docs: 0,
            processed_chunks: 0,
            current_phase: None,
            show_output: false,
        }
    }

    /// Start a new phase of processing
    pub fn start_phase(&mut self, phase: &str) {
        self.current_phase = Some(phase.to_string());
        if self.show_output {
            eprint!("  {}... ", phase);
            let _ = io::stderr().flush();
        }
    }

    /// Finish the current phase
    pub fn finish_phase(&mut self) {
        if self.show_output && self.current_phase.is_some() {
            eprintln!("done");
        }
        self.current_phase = None;
    }

    /// Update document progress
    pub fn update_docs(&mut self, count: usize) {
        self.processed_docs += count;
    }

    /// Update chunk progress
    pub fn update_chunks(&mut self, count: usize) {
        self.processed_chunks += count;
    }

    /// Get the number of processed documents
    pub fn docs_processed(&self) -> usize {
        self.processed_docs
    }

    /// Get the number of processed chunks
    pub fn chunks_processed(&self) -> usize {
        self.processed_chunks
    }

    /// Display final completion message
    pub fn complete(&self) {
        if self.show_output {
            eprintln!(
                "  Completed: {} docs, {} chunks",
                self.total_docs, self.processed_chunks
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracker() {
        let mut tracker = ProgressTracker::quiet(10);

        tracker.update_docs(5);
        tracker.update_chunks(50);

        assert_eq!(tracker.docs_processed(), 5);
        assert_eq!(tracker.chunks_processed(), 50);
    }

    #[test]
    fn test_phases() {
        let mut tracker = ProgressTracker::quiet(10);

        tracker.start_phase("Test phase");
        assert!(tracker.current_phase.is_some());

        tracker.finish_phase();
        assert!(tracker.current_phase.is_none());
    }
}

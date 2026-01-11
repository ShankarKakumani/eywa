//! The Provider Abstraction.
//!
//! This trait defines the standard interface for any LLM backend,
//! whether it's a cloud API (OpenAI) or a local model (Candle).

use anyhow::Result;
use async_trait::async_trait;
use super::types::{Message, CompletionResponse};

/// Metadata about a model's capabilities and costs.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub id: String,
    pub context_window: usize,
    // Add cost per token if needed later
}

/// The core trait for LLM interactions.
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Get the model's metadata (context window, ID, etc).
    fn metadata(&self) -> ModelMetadata;

    /// Estimate the number of tokens in a string.
    /// This is critical for context window management in RAG.
    fn count_tokens(&self, text: &str) -> Result<usize>;

    /// Send a chat completion request.
    async fn completion(&self, messages: &[Message]) -> Result<CompletionResponse>;
}

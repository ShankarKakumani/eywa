//! Context Window Management.
//!
//! This module handles the "scarce resource" of the LLM context window.
//! It is responsible for fitting the RAG chunks and user query into the
//! available token budget.

use anyhow::Result;
use super::provider::LLMProvider;
use super::types::Message;

/// Strategy for pruning context when it exceeds the limit.
#[derive(Debug, Clone, Copy)]
pub enum PruningStrategy {
    /// Keep the most recent messages.
    KeepRecent,
    /// Keep the instruction and query, but drop middle context.
    KeepEnds,
}

/// Manages the context window for a conversation.
pub struct WindowManager {
    /// The maximum number of tokens allowed associated with the model context window.
    pub context_limit: usize,
    /// Reserve some tokens for the answer (output buffer).
    pub output_buffer: usize,
}

impl WindowManager {
    pub fn new(context_limit: usize, output_buffer: usize) -> Self {
        Self {
            context_limit,
            output_buffer,
        }
    }

    /// Calculate available context for RAG chunks.
    ///
    /// available = context_limit - output_buffer - system_prompt - user_query - history
    pub fn available_tokens_for_context(
        &self,
        provider: &dyn LLMProvider,
        system_prompt: &str,
        query: &str,
        history: &[Message],
    ) -> Result<usize> {
        let system_tokens = provider.count_tokens(system_prompt)?;
        let query_tokens = provider.count_tokens(query)?;
        
        let mut history_tokens = 0;
        for msg in history {
            history_tokens += provider.count_tokens(&msg.content)?;
        }

        let used = system_tokens + query_tokens + history_tokens + self.output_buffer;
        
        if used >= self.context_limit {
            return Ok(0); // Context exhausted
        }

        Ok(self.context_limit - used)
    }

    /// Truncate a string to a specific token count.
    ///
    /// This is a simplified implementation. Real-world would decode/encode tokens.
    /// For now, we estimate char count (4 chars ~= 1 token).
    pub fn truncate(&self, _text: &str, _max_tokens: usize) -> String {
        // TODO: Implement real token-based truncation using the tokenizer
        // For now, return the whole string or implement simple char slicing
        // This requires access to the tokenizer which is in the provider.
        // We might need to refactor this to take the provider.
        todo!("Implement robust token truncation")
    }
}

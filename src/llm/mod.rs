//! LLM Layer: The Brain of Eywa
//!
//! This module handles all interactions with Large Language Models, including:
//! - Provider abstractions (OpenAI, Anthropic, Local)
//! - Context management (token counting, window pruning)
//! - Chat history and state

pub mod types;
pub mod provider;
pub mod context;
pub mod openai;
pub mod candle;

// Re-export key types
pub use types::{Message, Role, CompletionResponse};
pub use provider::LLMProvider;

use anyhow::{Result, anyhow};
use crate::config::{LLMConfig, LLMProviderType};

/// Factory to create an LLM provider instance from configuration.
pub async fn create_provider(config: &LLMConfig) -> Result<Box<dyn LLMProvider>> {
    match config.provider {
        LLMProviderType::OpenAI => {
            let api_key = config.api_key.clone()
                .ok_or_else(|| anyhow!("OpenAI API key not found in configuration"))?;
            let model = config.model.clone()
                .unwrap_or_else(|| "gpt-4o".to_string());
            
            Ok(Box::new(openai::OpenAIProvider::new(api_key, model)))
        }
        LLMProviderType::Local => {
            // Check if model is downloaded? For now let CandleProvider handle init failure
            Ok(Box::new(candle::CandleProvider::new().await?))
        }
        LLMProviderType::Anthropic => {
            Err(anyhow!("Anthropic provider not yet implemented"))
        }
        LLMProviderType::Disabled => {
            Err(anyhow!("LLM is disabled in settings"))
        }
        LLMProviderType::Ollama => {
            Err(anyhow!("Ollama provider not yet implemented"))
        }
    }
}

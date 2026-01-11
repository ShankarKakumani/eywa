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

// Re-export key types
pub use types::{Message, Role, CompletionResponse};
pub use provider::LLMProvider;

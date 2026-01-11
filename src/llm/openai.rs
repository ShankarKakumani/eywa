//! OpenAI API Provider.
//!
//! Implements the `LLMProvider` trait for OpenAI's Chat Completions API.

use super::provider::{LLMProvider, ModelMetadata};
use super::types::{CompletionResponse, Message, Role, Usage};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI Provider configuration and state.
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenAIProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn metadata(&self) -> ModelMetadata {
        let (window, _cost) = match self.model.as_str() {
            "gpt-4-turbo" | "gpt-4-turbo-preview" => (128_000, 10.0),
            "gpt-4" => (8_192, 30.0),
            "gpt-3.5-turbo" => (16_385, 0.5),
            _ => (4_096, 1.0), // Default safe assumption
        };

        ModelMetadata {
            id: self.model.clone(),
            context_window: window,
        }
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        // TODO: Use real tokenizer (tiktoken-rs)
        // For now, use the rule of thumb: 1 token ~= 4 characters
        Ok(text.len() / 4)
    }

    async fn completion(&self, messages: &[Message]) -> Result<CompletionResponse> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: messages.iter().map(ApiMessage::from).collect(),
            temperature: 0.7, // TODO: Make configurable
        };

        let response = self.client
            .post(OPENAI_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .context("Failed to send request to OpenAI")?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        let choice = chat_response.choices.first().context("No choices returned from OpenAI")?;
        
        Ok(CompletionResponse {
            content: choice.message.content.clone(),
            usage: chat_response.usage.into(),
        })
    }
}

// -----------------------------------------------------------------------------
// OpenAI DTOs (Data Transfer Objects)
// -----------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ApiMessage>,
    temperature: f32,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: String,
}

impl From<&Message> for ApiMessage {
    fn from(msg: &Message) -> Self {
        Self {
            role: match msg.role {
                Role::System => "system".to_string(),
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
            },
            content: msg.content.clone(),
        }
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: ApiUsage,
}

#[derive(Deserialize)]
struct Choice {
    message: ApiResponseMessage,
}

#[derive(Deserialize)]
struct ApiResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ApiUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl From<ApiUsage> for Usage {
    fn from(u: ApiUsage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

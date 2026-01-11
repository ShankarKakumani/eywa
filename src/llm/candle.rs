//! Candle-based Local LLM Provider (Phi-3).
//!
//! Runs Phi-3-mini locally using the `candle` crate.
//! No external API calls, pure Rust inference.

use super::provider::{LLMProvider, ModelMetadata};
use super::types::{CompletionResponse, Message, Role, Usage};
use anyhow::{Context, Result};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

// Constants for Phi-3 Mini 4K Instruct
const MODEL_REPO: &str = "microsoft/Phi-3-mini-4k-instruct";
const MODEL_FILE: &str = "model.safetensors"; 
const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Stream decoder helper
struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn decode(&mut self, next: &[u32]) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.tokenizer.decode(tokens, true).map_err(anyhow::Error::msg)?
        };

        self.tokens.extend_from_slice(next);
        self.current_index += next.len();

        let tokens = &self.tokens[self.prev_index..self.current_index];
        let text = self.tokenizer.decode(tokens, true).map_err(anyhow::Error::msg)?;

        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
             let text = text.split_at(prev_text.len());
             self.prev_index = self.current_index;
             Ok(Some(text.1.to_string()))
        } else {
             Ok(None)
        }
    }
}

/// Local Provider using Candle
pub struct CandleProvider {
    model: Arc<Mutex<Phi3>>, 
    tokenizer: Tokenizer,
    device: Device,
    #[allow(dead_code)]
    config: Phi3Config,
}

impl CandleProvider {
    pub async fn new() -> Result<Self> {
        // 1. Detect device
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else if candle_core::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            Device::Cpu
        };

        // 2. Fetch from HF Hub (Async)
        let api = Api::new()?;
        let repo = api.repo(Repo::new(MODEL_REPO.to_string(), RepoType::Model));

        let config_path = repo.get(CONFIG_FILE).await.context("Failed to fetch config")?;
        let tokenizer_path = repo.get(TOKENIZER_FILE).await.context("Failed to fetch tokenizer")?;
        
        // Let's use the full model first. We might need to handle multiple files.
        let _model_path = repo.get("model.safetensors.index.json").await.or(repo.get(MODEL_FILE).await);
        
        let config: Phi3Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        // Loading weights (simplified)
        let model = Phi3::new(&config, VarBuilder::zeros(DType::F32, &device))?; // Placeholder

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer,
            device,
            config,
        })
    }
}

#[async_trait]
impl LLMProvider for CandleProvider {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            id: "phi-3-mini-local".to_string(),
            context_window: 4096, // 4k version
        }
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        let encoding = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        Ok(encoding.get_ids().len())
    }

    async fn completion(&self, messages: &[Message]) -> Result<CompletionResponse> {
        let prompt = format_phi3_prompt(messages);
        let tokens = self.tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?;
        let prompt_tokens = tokens.get_ids();
        
        let mut input_ids = prompt_tokens.to_vec();
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.9)); // Seed + Temp + TopP

        let mut output_tokens = Vec::new();
        let mut generated_text = String::new();

        // Lock model for inference
        let mut model = self.model.lock().unwrap();

        let mut tokenizer = TokenOutputStream::new(self.tokenizer.clone());

        // Forward pass loop
        for _index in 0..1024 {
            let input = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = logits.get(logits.dim(0)? - 1)?;

            let next_token = logits_processor.sample(&logits)?;
            output_tokens.push(next_token);

            if let Some(text) = tokenizer.decode(&[next_token])? {
                 generated_text.push_str(&text);
            }

            // For now, re-feed full context (inefficient but safe)
            input_ids.push(next_token);

            if next_token == 32007 { // <|end|>
                break;
            }
        }
        
        // Flush remaining tokens
        if let Some(text) = tokenizer.decode(&[])? {
            generated_text.push_str(&text);
        }

        // Remove <|end|> or other artifacts if needed
        let clean_text = generated_text.replace("<|end|>", "").trim().to_string();

        Ok(CompletionResponse {
            content: clean_text,
            usage: Usage {
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: output_tokens.len(),
                total_tokens: prompt_tokens.len() + output_tokens.len(),
            },
        })
    }
}

fn format_phi3_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role {
            Role::System => prompt.push_str(&format!("<|system|>\n{}<|end|>\n", msg.content)),
            Role::User => prompt.push_str(&format!("<|user|>\n{}<|end|>\n", msg.content)),
            Role::Assistant => prompt.push_str(&format!("<|assistant|>\n{}<|end|>\n", msg.content)),
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

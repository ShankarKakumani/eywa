//! Local embeddings using Candle (pure Rust)
//!
//! Supports multiple embedding models configured via ~/.eywa/config.toml.
//! No ONNX runtime - pure Rust implementation.

use crate::config::{Config, EmbeddingModel};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimensions: usize,
}

impl Embedder {
    /// Create a new embedder using the model from config
    pub fn new() -> Result<Self> {
        let config = Config::load()?
            .ok_or_else(|| anyhow::anyhow!("Eywa not initialized. Run 'eywa' or 'eywa init' first."))?;
        Self::new_with_model(&config.embedding_model, true)
    }

    /// Create a new embedder with a specific model
    pub fn new_with_model(embedding_model: &EmbeddingModel, show_progress: bool) -> Result<Self> {
        let device = Device::Cpu;
        let model_id = embedding_model.hf_id();
        let dimensions = embedding_model.dimensions();

        if show_progress {
            eprintln!("  {} ({} MB)", embedding_model.name(), embedding_model.size_mb());
        }

        // Download model files from HuggingFace with progress
        let api = ApiBuilder::new()
            .with_progress(show_progress)
            .build()
            .context("Failed to create HuggingFace API")?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo.get("config.json").context("Failed to get config.json")?;
        let tokenizer_path = repo.get("tokenizer.json").context("Failed to get tokenizer.json")?;
        let weights_path = repo.get("model.safetensors").context("Failed to get model.safetensors")?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path)?;
        let bert_config: BertConfig = serde_json::from_str(&config_str)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)?
        };
        let model = BertModel::load(vb, &bert_config)?;

        if show_progress {
            eprintln!("done");
        }

        Ok(Self {
            model,
            tokenizer,
            device,
            dimensions,
        })
    }

    /// Create embedding for a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        Ok(embeddings.into_iter().next().unwrap())
    }

    /// Create embeddings for multiple texts
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // BERT models have max 512 position embeddings - must truncate
        const MAX_SEQ_LEN: usize = 512;

        let tokens = self.tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        // Cap at 512 tokens (model limit)
        let max_len = tokens.iter()
            .map(|t| t.get_ids().len().min(MAX_SEQ_LEN))
            .max()
            .unwrap_or(0);

        // Prepare input tensors
        let mut input_ids_vec = Vec::new();
        let mut attention_mask_vec = Vec::new();
        let mut token_type_ids_vec = Vec::new();

        for encoding in &tokens {
            // Truncate to MAX_SEQ_LEN tokens
            let ids: Vec<u32> = encoding.get_ids().iter().take(MAX_SEQ_LEN).copied().collect();
            let mask: Vec<u32> = encoding.get_attention_mask().iter().take(MAX_SEQ_LEN).copied().collect();

            let mut padded_ids = ids.clone();
            let mut padded_mask = mask.clone();
            let mut padded_types = vec![0u32; ids.len()];

            // Pad to max_len
            padded_ids.resize(max_len, 0);
            padded_mask.resize(max_len, 0);
            padded_types.resize(max_len, 0);

            input_ids_vec.extend(padded_ids);
            attention_mask_vec.extend(padded_mask);
            token_type_ids_vec.extend(padded_types);
        }

        let batch_size = texts.len();

        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &self.device)?;
        let attention_mask = Tensor::from_vec(attention_mask_vec, (batch_size, max_len), &self.device)?;
        let token_type_ids = Tensor::from_vec(token_type_ids_vec, (batch_size, max_len), &self.device)?;

        // Run model
        let embeddings = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling over sequence dimension
        let attention_mask_f = attention_mask.to_dtype(DTYPE)?;
        let mask_expanded = attention_mask_f.unsqueeze(2)?.broadcast_as(embeddings.shape())?;

        let sum_embeddings = (embeddings * mask_expanded)?.sum(1)?;
        let sum_mask = attention_mask_f.sum(1)?.unsqueeze(1)?;
        let mean_embeddings = sum_embeddings.broadcast_div(&sum_mask)?;

        // Normalize
        let norms = mean_embeddings.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?;
        let normalized = mean_embeddings.broadcast_div(&norms)?;

        // Convert to Vec<Vec<f32>>
        let embeddings_vec: Vec<Vec<f32>> = normalized.to_vec2()?;

        Ok(embeddings_vec)
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimensions
    }
}

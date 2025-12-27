//! Reranker using cross-encoder for better ranking
//!
//! Supports multiple reranker models configured via ~/.eywa/config.toml.

use crate::config::{Config, RerankerModel};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct Reranker {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Reranker {
    /// Create a new reranker using the model from config
    pub fn new() -> Result<Self> {
        let config = Config::load()?
            .ok_or_else(|| anyhow::anyhow!("Eywa not initialized. Run 'eywa' or 'eywa init' first."))?;
        Self::new_with_model(&config.reranker_model, true)
    }

    /// Create a new reranker with a specific model
    pub fn new_with_model(reranker_model: &RerankerModel, show_progress: bool) -> Result<Self> {
        let device = Device::Cpu;
        let model_id = reranker_model.hf_id();

        if show_progress {
            eprintln!("  {} ({} MB)", reranker_model.name(), reranker_model.size_mb());
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
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        let model = BertModel::load(vb, &bert_config)?;

        if show_progress {
            eprintln!("done");
        }

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Score query-document pairs
    /// Returns relevance scores (higher = more relevant)
    pub fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let mut scores = Vec::with_capacity(documents.len());

        // Process each (query, document) pair
        // Cross-encoder format: [CLS] query [SEP] document [SEP]
        for doc in documents {
            let score = self.score_pair(query, doc)?;
            scores.push(score);
        }

        Ok(scores)
    }

    /// Score a single query-document pair
    fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        // Tokenize as a pair
        let encoding = self.tokenizer
            .encode((query, document), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();

        let len = ids.len();

        let input_ids = Tensor::from_vec(ids.to_vec(), (1, len), &self.device)?;
        let attention_mask = Tensor::from_vec(mask.to_vec(), (1, len), &self.device)?;
        let token_type_ids = Tensor::from_vec(type_ids.to_vec(), (1, len), &self.device)?;

        // Run model
        let output = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Get [CLS] token output (first token)
        let cls_output = output.i((0, 0))?;

        // For reranker, we typically use a linear layer on top of [CLS]
        // But bge-reranker outputs the score directly from the first dimension
        // Take the first value as the relevance score
        let score: f32 = cls_output.i(0)?.to_scalar()?;

        // Apply sigmoid to get score between 0 and 1
        let score = 1.0 / (1.0 + (-score).exp());

        Ok(score)
    }

    /// Rerank search results and return sorted by reranker score
    pub fn rerank_results<T: Clone>(
        &self,
        query: &str,
        results: Vec<(T, String)>, // (item, content)
        top_k: usize,
    ) -> Result<Vec<(T, f32)>> {
        if results.is_empty() {
            return Ok(vec![]);
        }

        let documents: Vec<String> = results.iter().map(|(_, content)| content.clone()).collect();
        let scores = self.rerank(query, &documents)?;

        // Combine items with scores
        let mut scored: Vec<(T, f32)> = results
            .into_iter()
            .zip(scores)
            .map(|((item, _), score)| (item, score))
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top K
        Ok(scored.into_iter().take(top_k).collect())
    }
}

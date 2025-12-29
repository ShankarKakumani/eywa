//! Configuration management for Eywa
//!
//! Handles model selection and persistence of user preferences.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Device preference for compute
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum DevicePreference {
    /// Automatically detect best available device (GPU if available, else CPU)
    #[default]
    Auto,
    /// Force CPU usage
    Cpu,
    /// Force Metal GPU (macOS Apple Silicon)
    Metal,
    /// Force CUDA GPU (NVIDIA)
    Cuda,
}

impl DevicePreference {
    /// Display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// Get all available options
    pub fn all() -> Vec<Self> {
        vec![Self::Auto, Self::Cpu, Self::Metal, Self::Cuda]
    }
}

/// Available embedding models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingModel {
    /// BGE Base - balanced quality and size
    BgeBaseEnV15,
    /// BGE Small - faster, smaller footprint
    BgeSmallEnV15,
    /// Nomic Embed - high quality, larger
    NomicEmbedTextV15,
    /// all-MiniLM-L6-v2 - 6 layers, smallest/fastest
    AllMiniLmL6V2,
    /// all-MiniLM-L12-v2 - 12 layers, better quality than L6 (default)
    AllMiniLmL12V2,
}

impl EmbeddingModel {
    /// Display name for the model
    pub fn name(&self) -> &'static str {
        match self {
            Self::BgeBaseEnV15 => "bge-base-en-v1.5",
            Self::BgeSmallEnV15 => "bge-small-en-v1.5",
            Self::NomicEmbedTextV15 => "nomic-embed-text-v1.5",
            Self::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            Self::AllMiniLmL12V2 => "all-MiniLM-L12-v2",
        }
    }

    /// HuggingFace model ID
    pub fn hf_id(&self) -> &'static str {
        match self {
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::NomicEmbedTextV15 => "nomic-ai/nomic-embed-text-v1.5",
            Self::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::AllMiniLmL12V2 => "sentence-transformers/all-MiniLM-L12-v2",
        }
    }

    /// Embedding dimensions
    pub fn dimensions(&self) -> usize {
        match self {
            Self::BgeBaseEnV15 => 768,
            Self::BgeSmallEnV15 => 384,
            Self::NomicEmbedTextV15 => 768,
            Self::AllMiniLmL6V2 => 384,
            Self::AllMiniLmL12V2 => 384,
        }
    }

    /// Approximate model size in MB
    pub fn size_mb(&self) -> u32 {
        match self {
            Self::BgeBaseEnV15 => 418,
            Self::BgeSmallEnV15 => 134,
            Self::NomicEmbedTextV15 => 548,
            Self::AllMiniLmL6V2 => 86,
            Self::AllMiniLmL12V2 => 134,
        }
    }

    /// Get all available models
    pub fn all() -> Vec<Self> {
        vec![
            Self::BgeBaseEnV15,
            Self::BgeSmallEnV15,
            Self::NomicEmbedTextV15,
            Self::AllMiniLmL6V2,
            Self::AllMiniLmL12V2,
        ]
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self::AllMiniLmL12V2
    }
}

/// Available reranker models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RerankerModel {
    /// Jina v2 multilingual - code-aware, multilingual (default)
    JinaRerankerV2BaseMultilingual,
    /// Jina v1 turbo - faster, English only
    JinaRerankerV1TurboEn,
    /// BGE reranker - balanced
    BgeRerankerBase,
    /// MS-MARCO MiniLM - legacy, smallest
    MsMarcoMiniLmL6V2,
}

impl RerankerModel {
    /// Display name for the model
    pub fn name(&self) -> &'static str {
        match self {
            Self::JinaRerankerV2BaseMultilingual => "jina-reranker-v2-base-multilingual",
            Self::JinaRerankerV1TurboEn => "jina-reranker-v1-turbo-en",
            Self::BgeRerankerBase => "bge-reranker-base",
            Self::MsMarcoMiniLmL6V2 => "ms-marco-MiniLM-L-6-v2",
        }
    }

    /// HuggingFace model ID
    pub fn hf_id(&self) -> &'static str {
        match self {
            Self::JinaRerankerV2BaseMultilingual => "jinaai/jina-reranker-v2-base-multilingual",
            Self::JinaRerankerV1TurboEn => "jinaai/jina-reranker-v1-turbo-en",
            Self::BgeRerankerBase => "BAAI/bge-reranker-base",
            Self::MsMarcoMiniLmL6V2 => "cross-encoder/ms-marco-MiniLM-L-6-v2",
        }
    }

    /// Approximate model size in MB
    pub fn size_mb(&self) -> u32 {
        match self {
            Self::JinaRerankerV2BaseMultilingual => 278,
            Self::JinaRerankerV1TurboEn => 100,
            Self::BgeRerankerBase => 278,
            Self::MsMarcoMiniLmL6V2 => 86,
        }
    }

    /// Get all available models
    pub fn all() -> Vec<Self> {
        vec![
            Self::JinaRerankerV2BaseMultilingual,
            Self::JinaRerankerV1TurboEn,
            Self::BgeRerankerBase,
            Self::MsMarcoMiniLmL6V2,
        ]
    }
}

impl Default for RerankerModel {
    fn default() -> Self {
        Self::MsMarcoMiniLmL6V2 // BERT-based, works with Candle
    }
}

/// Eywa configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Selected embedding model
    pub embedding_model: EmbeddingModel,
    /// Selected reranker model
    pub reranker_model: RerankerModel,
    /// Device preference (auto, cpu, metal, cuda)
    #[serde(default)]
    pub device: DevicePreference,
    /// Version of config schema (for future migrations)
    #[serde(default = "default_version")]
    pub version: u32,
}

fn default_version() -> u32 {
    1
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding_model: EmbeddingModel::default(),
            reranker_model: RerankerModel::default(),
            device: DevicePreference::default(),
            version: 1,
        }
    }
}

impl Config {
    /// Get the config file path (~/.eywa/config.toml)
    pub fn path() -> Result<PathBuf> {
        let home = std::env::var("HOME").context("HOME environment variable not set")?;
        Ok(PathBuf::from(home).join(".eywa").join("config.toml"))
    }

    /// Check if config exists (i.e., not first run)
    pub fn exists() -> bool {
        Self::path().map(|p| p.exists()).unwrap_or(false)
    }

    /// Load config from disk, or return None if it doesn't exist
    pub fn load() -> Result<Option<Self>> {
        let path = Self::path()?;
        if !path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(&path)
            .context("Failed to read config file")?;
        let config: Self = toml::from_str(&content)
            .context("Failed to parse config file")?;
        Ok(Some(config))
    }

    /// Save config to disk
    pub fn save(&self) -> Result<()> {
        let path = Self::path()?;

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create config directory")?;
        }

        let content = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;
        std::fs::write(&path, content)
            .context("Failed to write config file")?;

        Ok(())
    }

    /// Get total download size for selected models
    pub fn total_download_size_mb(&self) -> u32 {
        self.embedding_model.size_mb() + self.reranker_model.size_mb()
    }
}

/// Get the data directory path (~/.eywa/data)
pub fn data_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME environment variable not set")?;
    Ok(PathBuf::from(home).join(".eywa").join("data"))
}

/// Get the base eywa directory path (~/.eywa)
pub fn eywa_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME environment variable not set")?;
    Ok(PathBuf::from(home).join(".eywa"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.embedding_model, EmbeddingModel::AllMiniLmL12V2);
        assert_eq!(config.reranker_model, RerankerModel::MsMarcoMiniLmL6V2);
    }

    #[test]
    fn test_model_metadata() {
        let model = EmbeddingModel::BgeBaseEnV15;
        assert_eq!(model.dimensions(), 768);
        assert_eq!(model.size_mb(), 418);
        assert_eq!(model.hf_id(), "BAAI/bge-base-en-v1.5");
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.embedding_model, parsed.embedding_model);
        assert_eq!(config.reranker_model, parsed.reranker_model);
    }
}

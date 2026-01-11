//! Configuration management for Eywa
//!
//! Handles model selection and persistence of user preferences.
//! Supports both curated models and custom HuggingFace models.

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

// ─────────────────────────────────────────────────────────────────────────────
// Embedding Model Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingModelConfig {
    /// Unique identifier (used in config file)
    pub id: String,
    /// Display name
    pub name: String,
    /// HuggingFace repository ID
    pub repo_id: String,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Approximate model size in MB
    #[serde(default)]
    pub size_mb: u32,
    /// Whether this is a curated (built-in) model
    #[serde(default)]
    pub curated: bool,
}

impl EmbeddingModelConfig {
    /// Create a custom model config
    pub fn custom(repo_id: &str, dimensions: usize) -> Self {
        let name = repo_id.split('/').last().unwrap_or(repo_id).to_string();
        Self {
            id: format!("custom:{}", repo_id),
            name,
            repo_id: repo_id.to_string(),
            dimensions,
            size_mb: 0,
            curated: false,
        }
    }

    /// HuggingFace model ID (for compatibility with existing code)
    pub fn hf_id(&self) -> &str {
        &self.repo_id
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Curated Models
    // ─────────────────────────────────────────────────────────────────────────

    pub fn all_minilm_l6_v2() -> Self {
        Self {
            id: "all-MiniLM-L6-v2".to_string(),
            name: "all-MiniLM-L6-v2".to_string(),
            repo_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimensions: 384,
            size_mb: 86,
            curated: true,
        }
    }

    pub fn all_minilm_l12_v2() -> Self {
        Self {
            id: "all-MiniLM-L12-v2".to_string(),
            name: "all-MiniLM-L12-v2".to_string(),
            repo_id: "sentence-transformers/all-MiniLM-L12-v2".to_string(),
            dimensions: 384,
            size_mb: 134,
            curated: true,
        }
    }

    pub fn bge_small_en_v15() -> Self {
        Self {
            id: "bge-small-en-v1.5".to_string(),
            name: "bge-small-en-v1.5".to_string(),
            repo_id: "BAAI/bge-small-en-v1.5".to_string(),
            dimensions: 384,
            size_mb: 134,
            curated: true,
        }
    }

    pub fn bge_base_en_v15() -> Self {
        Self {
            id: "bge-base-en-v1.5".to_string(),
            name: "bge-base-en-v1.5".to_string(),
            repo_id: "BAAI/bge-base-en-v1.5".to_string(),
            dimensions: 768,
            size_mb: 418,
            curated: true,
        }
    }

    pub fn nomic_embed_text_v15() -> Self {
        Self {
            id: "nomic-embed-text-v1.5".to_string(),
            name: "nomic-embed-text-v1.5".to_string(),
            repo_id: "nomic-ai/nomic-embed-text-v1.5".to_string(),
            dimensions: 768,
            size_mb: 548,
            curated: true,
        }
    }

    /// Get all curated embedding models
    pub fn curated_models() -> Vec<Self> {
        vec![
            Self::all_minilm_l6_v2(),
            Self::all_minilm_l12_v2(),
            Self::bge_small_en_v15(),
            Self::bge_base_en_v15(),
            Self::nomic_embed_text_v15(),
        ]
    }

    /// Find a curated model by ID
    pub fn find_curated(id: &str) -> Option<Self> {
        Self::curated_models().into_iter().find(|m| m.id == id)
    }
}

impl Default for EmbeddingModelConfig {
    fn default() -> Self {
        Self::all_minilm_l12_v2()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reranker Model Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Reranker model configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RerankerModelConfig {
    /// Unique identifier (used in config file)
    pub id: String,
    /// Display name
    pub name: String,
    /// HuggingFace repository ID
    pub repo_id: String,
    /// Approximate model size in MB
    #[serde(default)]
    pub size_mb: u32,
    /// Whether this is a curated (built-in) model
    #[serde(default)]
    pub curated: bool,
}

impl RerankerModelConfig {
    /// Create a custom model config
    pub fn custom(repo_id: &str) -> Self {
        let name = repo_id.split('/').last().unwrap_or(repo_id).to_string();
        Self {
            id: format!("custom:{}", repo_id),
            name,
            repo_id: repo_id.to_string(),
            size_mb: 0,
            curated: false,
        }
    }

    /// HuggingFace model ID (for compatibility with existing code)
    pub fn hf_id(&self) -> &str {
        &self.repo_id
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Curated Models
    // ─────────────────────────────────────────────────────────────────────────

    pub fn ms_marco_minilm_l6_v2() -> Self {
        Self {
            id: "ms-marco-MiniLM-L-6-v2".to_string(),
            name: "ms-marco-MiniLM-L-6-v2".to_string(),
            repo_id: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            size_mb: 86,
            curated: true,
        }
    }

    pub fn bge_reranker_base() -> Self {
        Self {
            id: "bge-reranker-base".to_string(),
            name: "bge-reranker-base".to_string(),
            repo_id: "BAAI/bge-reranker-base".to_string(),
            size_mb: 278,
            curated: true,
        }
    }

    pub fn jina_reranker_v1_turbo_en() -> Self {
        Self {
            id: "jina-reranker-v1-turbo-en".to_string(),
            name: "jina-reranker-v1-turbo-en".to_string(),
            repo_id: "jinaai/jina-reranker-v1-turbo-en".to_string(),
            size_mb: 100,
            curated: true,
        }
    }

    pub fn jina_reranker_v2_base_multilingual() -> Self {
        Self {
            id: "jina-reranker-v2-base-multilingual".to_string(),
            name: "jina-reranker-v2-base-multilingual".to_string(),
            repo_id: "jinaai/jina-reranker-v2-base-multilingual".to_string(),
            size_mb: 278,
            curated: true,
        }
    }

    /// Get all curated reranker models
    pub fn curated_models() -> Vec<Self> {
        vec![
            Self::ms_marco_minilm_l6_v2(),
            Self::bge_reranker_base(),
            Self::jina_reranker_v1_turbo_en(),
            Self::jina_reranker_v2_base_multilingual(),
        ]
    }

    /// Find a curated model by ID
    pub fn find_curated(id: &str) -> Option<Self> {
        Self::curated_models().into_iter().find(|m| m.id == id)
    }
}

impl Default for RerankerModelConfig {
    fn default() -> Self {
        Self::ms_marco_minilm_l6_v2()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy Enum Types (for backward compatibility)
// ─────────────────────────────────────────────────────────────────────────────

/// Legacy embedding model enum (for config migration)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingModel {
    BgeBaseEnV15,
    BgeSmallEnV15,
    NomicEmbedTextV15,
    AllMiniLmL6V2,
    AllMiniLmL12V2,
}

impl EmbeddingModel {
    pub fn to_config(&self) -> EmbeddingModelConfig {
        match self {
            Self::BgeBaseEnV15 => EmbeddingModelConfig::bge_base_en_v15(),
            Self::BgeSmallEnV15 => EmbeddingModelConfig::bge_small_en_v15(),
            Self::NomicEmbedTextV15 => EmbeddingModelConfig::nomic_embed_text_v15(),
            Self::AllMiniLmL6V2 => EmbeddingModelConfig::all_minilm_l6_v2(),
            Self::AllMiniLmL12V2 => EmbeddingModelConfig::all_minilm_l12_v2(),
        }
    }

    // Legacy methods for compatibility
    pub fn name(&self) -> &'static str {
        match self {
            Self::BgeBaseEnV15 => "bge-base-en-v1.5",
            Self::BgeSmallEnV15 => "bge-small-en-v1.5",
            Self::NomicEmbedTextV15 => "nomic-embed-text-v1.5",
            Self::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            Self::AllMiniLmL12V2 => "all-MiniLM-L12-v2",
        }
    }

    pub fn hf_id(&self) -> &'static str {
        match self {
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::NomicEmbedTextV15 => "nomic-ai/nomic-embed-text-v1.5",
            Self::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::AllMiniLmL12V2 => "sentence-transformers/all-MiniLM-L12-v2",
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            Self::BgeBaseEnV15 => 768,
            Self::BgeSmallEnV15 => 384,
            Self::NomicEmbedTextV15 => 768,
            Self::AllMiniLmL6V2 => 384,
            Self::AllMiniLmL12V2 => 384,
        }
    }

    pub fn size_mb(&self) -> u32 {
        match self {
            Self::BgeBaseEnV15 => 418,
            Self::BgeSmallEnV15 => 134,
            Self::NomicEmbedTextV15 => 548,
            Self::AllMiniLmL6V2 => 86,
            Self::AllMiniLmL12V2 => 134,
        }
    }

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

/// Legacy reranker model enum (for config migration)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RerankerModel {
    JinaRerankerV2BaseMultilingual,
    JinaRerankerV1TurboEn,
    BgeRerankerBase,
    MsMarcoMiniLmL6V2,
}

impl RerankerModel {
    pub fn to_config(&self) -> RerankerModelConfig {
        match self {
            Self::JinaRerankerV2BaseMultilingual => RerankerModelConfig::jina_reranker_v2_base_multilingual(),
            Self::JinaRerankerV1TurboEn => RerankerModelConfig::jina_reranker_v1_turbo_en(),
            Self::BgeRerankerBase => RerankerModelConfig::bge_reranker_base(),
            Self::MsMarcoMiniLmL6V2 => RerankerModelConfig::ms_marco_minilm_l6_v2(),
        }
    }

    // Legacy methods for compatibility
    pub fn name(&self) -> &'static str {
        match self {
            Self::JinaRerankerV2BaseMultilingual => "jina-reranker-v2-base-multilingual",
            Self::JinaRerankerV1TurboEn => "jina-reranker-v1-turbo-en",
            Self::BgeRerankerBase => "bge-reranker-base",
            Self::MsMarcoMiniLmL6V2 => "ms-marco-MiniLM-L-6-v2",
        }
    }

    pub fn hf_id(&self) -> &'static str {
        match self {
            Self::JinaRerankerV2BaseMultilingual => "jinaai/jina-reranker-v2-base-multilingual",
            Self::JinaRerankerV1TurboEn => "jinaai/jina-reranker-v1-turbo-en",
            Self::BgeRerankerBase => "BAAI/bge-reranker-base",
            Self::MsMarcoMiniLmL6V2 => "cross-encoder/ms-marco-MiniLM-L-6-v2",
        }
    }

    pub fn size_mb(&self) -> u32 {
        match self {
            Self::JinaRerankerV2BaseMultilingual => 278,
            Self::JinaRerankerV1TurboEn => 100,
            Self::BgeRerankerBase => 278,
            Self::MsMarcoMiniLmL6V2 => 86,
        }
    }

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
        Self::MsMarcoMiniLmL6V2
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration (supports both legacy and new format)
// ─────────────────────────────────────────────────────────────────────────────

/// Legacy config format (v1)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyConfig {
    embedding_model: EmbeddingModel,
    reranker_model: RerankerModel,
    #[serde(default)]
    device: DevicePreference,
    #[serde(default = "default_version")]
    version: u32,
}

/// Eywa configuration (v2 - struct-based models)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Selected embedding model
    pub embedding_model: EmbeddingModelConfig,
    /// Selected reranker model
    pub reranker_model: RerankerModelConfig,
    /// Device preference (auto, cpu, metal, cuda)
    #[serde(default)]
    pub device: DevicePreference,
    /// Version of config schema
    #[serde(default = "current_version")]
    pub version: u32,
    /// LLM Configuration
    #[serde(default)]
    pub llm: LLMConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LLMConfig {
    /// LLM Provider (openai, local, etc.)
    #[serde(default)]
    pub provider: LLMProviderType,
    /// Model name (e.g. gpt-4o, phi-3-mini)
    #[serde(default)]
    pub model: Option<String>,
    /// API Key (for cloud providers)
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum LLMProviderType {
    #[default]
    OpenAI,
    Local,
    Anthropic,
    Ollama,
    Disabled,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: LLMProviderType::OpenAI,
            model: None,
            api_key: None,
        }
    }
}

fn default_version() -> u32 {
    1
}

fn current_version() -> u32 {
    2
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding_model: EmbeddingModelConfig::default(),
            reranker_model: RerankerModelConfig::default(),
            device: DevicePreference::default(),
            version: current_version(),
            llm: LLMConfig::default(),
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
    /// Automatically migrates legacy v1 configs to v2 format
    pub fn load() -> Result<Option<Self>> {
        let path = Self::path()?;
        if !path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(&path)
            .context("Failed to read config file")?;

        // Try parsing as v2 config first
        if let Ok(config) = toml::from_str::<Config>(&content) {
            if config.version >= 2 {
                return Ok(Some(config));
            }
        }

        // Try parsing as legacy v1 config and migrate
        if let Ok(legacy) = toml::from_str::<LegacyConfig>(&content) {
            let migrated = Config {
                embedding_model: legacy.embedding_model.to_config(),
                reranker_model: legacy.reranker_model.to_config(),
                device: legacy.device,
                version: current_version(),
                llm: LLMConfig::default(),
            };
            // Save migrated config
            if let Err(e) = migrated.save() {
                eprintln!("Warning: Failed to save migrated config: {}", e);
            }
            return Ok(Some(migrated));
        }

        // If both fail, return error
        let config: Config = toml::from_str(&content)
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
        self.embedding_model.size_mb + self.reranker_model.size_mb
    }

    /// Update embedding model
    pub fn set_embedding_model(&mut self, model: EmbeddingModelConfig) {
        self.embedding_model = model;
    }

    /// Update reranker model
    pub fn set_reranker_model(&mut self, model: RerankerModelConfig) {
        self.reranker_model = model;
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
        assert_eq!(config.embedding_model.id, "all-MiniLM-L12-v2");
        assert_eq!(config.reranker_model.id, "ms-marco-MiniLM-L-6-v2");
        assert_eq!(config.version, 2);
    }

    #[test]
    fn test_curated_models() {
        let embedders = EmbeddingModelConfig::curated_models();
        assert_eq!(embedders.len(), 5);

        let rerankers = RerankerModelConfig::curated_models();
        assert_eq!(rerankers.len(), 4);
    }

    #[test]
    fn test_custom_model() {
        let model = EmbeddingModelConfig::custom("sentence-transformers/all-mpnet-base-v2", 768);
        assert_eq!(model.repo_id, "sentence-transformers/all-mpnet-base-v2");
        assert_eq!(model.dimensions, 768);
        assert!(!model.curated);
    }

    #[test]
    fn test_find_curated() {
        let model = EmbeddingModelConfig::find_curated("bge-base-en-v1.5");
        assert!(model.is_some());
        assert_eq!(model.unwrap().dimensions, 768);

        let not_found = EmbeddingModelConfig::find_curated("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.embedding_model.id, parsed.embedding_model.id);
        assert_eq!(config.reranker_model.id, parsed.reranker_model.id);
    }

    #[test]
    fn test_legacy_conversion() {
        let legacy = EmbeddingModel::BgeBaseEnV15;
        let config = legacy.to_config();
        assert_eq!(config.id, "bge-base-en-v1.5");
        assert_eq!(config.dimensions, 768);
    }
}

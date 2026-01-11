//! Custom HuggingFace model downloader with progress callbacks
//!
//! Downloads model files with streaming progress reporting for TUI visualization.
//! Fully compatible with hf-hub cache structure.

use crate::config::{EmbeddingModel, EmbeddingModelConfig, RerankerModel, RerankerModelConfig};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use std::path::PathBuf;

/// Files required for each model
const MODEL_FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

/// Progress update from a download
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub file_name: String,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub done: bool,
}

/// Model download task
#[derive(Debug, Clone)]
pub struct ModelTask {
    pub name: String,
    pub repo_id: String,
    pub size_mb: u32,
    pub files: Vec<FileTask>,
    pub commit_hash: Option<String>,
}

/// Individual file download task
#[derive(Debug, Clone)]
pub struct FileTask {
    pub name: String,
    pub url: String,
    pub cache_path: PathBuf,
    pub size_bytes: Option<u64>,
    pub downloaded_bytes: u64,
    pub done: bool,
}

/// Downloader for HuggingFace models
pub struct ModelDownloader {
    client: reqwest::Client,
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new() -> Self {
        let cache_dir = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".cache")
            .join("huggingface")
            .join("hub");

        Self {
            client: reqwest::Client::new(),
            cache_dir,
        }
    }

    /// Check if a model is already cached
    pub fn is_cached<M: ModelInfo>(&self, model: &M) -> bool {
        let model_dir = self.model_cache_dir(model.hf_id());
        MODEL_FILES.iter().all(|file| {
            self.find_cached_file(&model_dir, file).is_some()
        })
    }

    /// Create download tasks for a model (async to fetch commit hash)
    pub async fn create_tasks<M: ModelInfo>(&self, model: &M) -> Result<ModelTask> {
        let repo_id = model.hf_id().to_string();
        let model_dir = self.model_cache_dir(&repo_id);

        // Check if we have a cached commit hash
        let cached_commit = self.get_cached_commit(&model_dir);

        // Try to get commit hash from HuggingFace API
        let commit_hash = match self.fetch_commit_hash(&repo_id).await {
            Ok(hash) => Some(hash),
            Err(_) => cached_commit.clone(),
        };

        let files: Vec<FileTask> = MODEL_FILES
            .iter()
            .map(|file| {
                // Check if file exists in any snapshot
                let cached = self.find_cached_file(&model_dir, file);
                let is_done = cached.is_some();

                // Determine cache path based on commit hash
                let cache_path = if let Some(ref hash) = commit_hash {
                    model_dir.join("snapshots").join(hash).join(file)
                } else {
                    // Fallback if we couldn't get commit hash
                    model_dir.join("snapshots").join("main").join(file)
                };

                // Get size from cached file if available
                let (downloaded_bytes, size_bytes) = if let Some(ref cached_path) = cached {
                    let size = std::fs::metadata(cached_path).map(|m| m.len()).unwrap_or(0);
                    (size, Some(size))
                } else {
                    (0, None)
                };

                FileTask {
                    name: file.to_string(),
                    url: format!(
                        "https://huggingface.co/{}/resolve/main/{}",
                        repo_id, file
                    ),
                    cache_path,
                    size_bytes,
                    downloaded_bytes,
                    done: is_done,
                }
            })
            .collect();

        Ok(ModelTask {
            name: model.name().to_string(),
            repo_id,
            size_mb: model.size_mb(),
            files,
            commit_hash,
        })
    }

    /// Fetch the current commit hash for a model from HuggingFace API
    async fn fetch_commit_hash(&self, repo_id: &str) -> Result<String> {
        let url = format!("https://huggingface.co/api/models/{}", repo_id);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch model info")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to fetch model info: HTTP {}", response.status());
        }

        let json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse model info")?;

        json.get("sha")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("No sha field in model info"))
    }

    /// Get cached commit hash from refs/main
    fn get_cached_commit(&self, model_dir: &PathBuf) -> Option<String> {
        let refs_path = model_dir.join("refs").join("main");
        std::fs::read_to_string(refs_path).ok().map(|s| s.trim().to_string())
    }

    /// Save commit hash to refs/main
    fn save_commit_ref(&self, model_dir: &PathBuf, commit_hash: &str) -> Result<()> {
        let refs_dir = model_dir.join("refs");
        std::fs::create_dir_all(&refs_dir)?;
        std::fs::write(refs_dir.join("main"), commit_hash)?;
        Ok(())
    }

    /// Download a file with progress callback
    pub async fn download_file<F>(
        &self,
        task: &mut FileTask,
        model_dir: &PathBuf,
        commit_hash: Option<&str>,
        on_progress: F,
    ) -> Result<()>
    where
        F: Fn(DownloadProgress) + Send,
    {
        if task.done {
            on_progress(DownloadProgress {
                file_name: task.name.clone(),
                bytes_downloaded: task.downloaded_bytes,
                total_bytes: task.size_bytes,
                done: true,
            });
            return Ok(());
        }

        // Create cache directory structure
        if let Some(parent) = task.cache_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create cache directory")?;
        }

        // Save commit ref if we have one
        if let Some(hash) = commit_hash {
            self.save_commit_ref(model_dir, hash)?;
        }

        // Start download
        let response = self
            .client
            .get(&task.url)
            .send()
            .await
            .context("Failed to start download")?;

        if !response.status().is_success() {
            anyhow::bail!("Download failed: HTTP {}", response.status());
        }

        let total_size = response.content_length();
        task.size_bytes = total_size;

        // Create temp file for atomic write
        let temp_path = task.cache_path.with_extension("tmp");
        let mut file = tokio::fs::File::create(&temp_path)
            .await
            .context("Failed to create temp file")?;

        // Stream the download
        let mut stream = response.bytes_stream();
        let mut downloaded: u64 = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error reading download stream")?;
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
                .await
                .context("Failed to write chunk")?;

            downloaded += chunk.len() as u64;
            task.downloaded_bytes = downloaded;

            on_progress(DownloadProgress {
                file_name: task.name.clone(),
                bytes_downloaded: downloaded,
                total_bytes: total_size,
                done: false,
            });
        }

        // Flush and close
        tokio::io::AsyncWriteExt::flush(&mut file).await?;
        drop(file);

        // Atomic rename
        std::fs::rename(&temp_path, &task.cache_path)
            .context("Failed to finalize download")?;

        task.done = true;

        on_progress(DownloadProgress {
            file_name: task.name.clone(),
            bytes_downloaded: downloaded,
            total_bytes: total_size,
            done: true,
        });

        Ok(())
    }

    /// Get the cache directory for a model
    pub fn model_cache_dir(&self, repo_id: &str) -> PathBuf {
        let dir_name = format!("models--{}", repo_id.replace('/', "--"));
        self.cache_dir.join(dir_name)
    }

    /// Delete a cached model from disk
    pub fn delete_cached<M: ModelInfo>(&self, model: &M) -> Result<()> {
        let model_dir = self.model_cache_dir(model.hf_id());
        if model_dir.exists() {
            std::fs::remove_dir_all(&model_dir)
                .context(format!("Failed to delete model cache at {:?}", model_dir))?;
        }
        Ok(())
    }

    /// Find a cached file (checks all snapshots)
    fn find_cached_file(&self, model_dir: &PathBuf, file: &str) -> Option<PathBuf> {
        let snapshots_dir = model_dir.join("snapshots");
        if !snapshots_dir.exists() {
            return None;
        }

        // Check each snapshot directory
        if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
            for entry in entries.flatten() {
                let file_path = entry.path().join(file);
                if file_path.exists() {
                    return Some(file_path);
                }
            }
        }

        None
    }
}

/// Trait for model info (implemented by model config structs)
pub trait ModelInfo {
    fn name(&self) -> &str;
    fn hf_id(&self) -> &str;
    fn size_mb(&self) -> u32;
}

// Implementation for new config structs
impl ModelInfo for EmbeddingModelConfig {
    fn name(&self) -> &str {
        &self.name
    }

    fn hf_id(&self) -> &str {
        &self.repo_id
    }

    fn size_mb(&self) -> u32 {
        self.size_mb
    }
}

impl ModelInfo for RerankerModelConfig {
    fn name(&self) -> &str {
        &self.name
    }

    fn hf_id(&self) -> &str {
        &self.repo_id
    }

    fn size_mb(&self) -> u32 {
        self.size_mb
    }
}

// Legacy implementations (for backward compatibility)
impl ModelInfo for EmbeddingModel {
    fn name(&self) -> &str {
        EmbeddingModel::name(self)
    }

    fn hf_id(&self) -> &str {
        EmbeddingModel::hf_id(self)
    }

    fn size_mb(&self) -> u32 {
        EmbeddingModel::size_mb(self)
    }
}

impl ModelInfo for RerankerModel {
    fn name(&self) -> &str {
        RerankerModel::name(self)
    }

    fn hf_id(&self) -> &str {
        RerankerModel::hf_id(self)
    }

    fn size_mb(&self) -> u32 {
        RerankerModel::size_mb(self)
    }
}

pub struct Phi3Model;

impl Phi3Model {
    pub fn new() -> Self {
        Self
    }
}

impl ModelInfo for Phi3Model {
    fn name(&self) -> &str {
        "Phi-3-mini-4k-instruct"
    }

    fn hf_id(&self) -> &str {
        "microsoft/Phi-3-mini-4k-instruct"
    }

    fn size_mb(&self) -> u32 {
        2200 // Approx 2.2GB
    }
}

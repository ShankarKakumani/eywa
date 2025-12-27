//! Setup wizard with TUI for first-run experience
//!
//! Provides a polished download experience with progress visualization.

mod download;
mod tui;

use crate::config::Config;
use anyhow::Result;

pub use download::ModelDownloader;
pub use tui::SetupWizard;

/// Run the setup wizard TUI for downloading models
///
/// This takes over the terminal, shows a beautiful dashboard with
/// download progress, and exits when complete.
pub fn run_download_wizard(config: &Config) -> Result<()> {
    let mut wizard = SetupWizard::new(config.clone())?;
    wizard.run()
}

/// Check if models are already downloaded (in HF cache)
pub fn models_cached(config: &Config) -> bool {
    let downloader = ModelDownloader::new();
    downloader.is_cached(&config.embedding_model) && downloader.is_cached(&config.reranker_model)
}

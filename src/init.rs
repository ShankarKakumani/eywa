//! Initialization flow for Eywa
//!
//! Handles first-run setup and model selection.

use crate::config::{Config, DevicePreference, EmbeddingModelConfig, RerankerModelConfig};
use anyhow::Result;
use std::io::{self, Write};

/// Result of running the init flow
#[derive(Debug)]
pub enum InitResult {
    /// User completed init with this config
    Configured(Config),
    /// User cancelled the init
    Cancelled,
}

/// Run the interactive init flow
pub async fn run_init(existing_config: Option<&Config>) -> Result<InitResult> {
    let is_reinit = existing_config.is_some();

    if is_reinit {
        println!("\nCurrent configuration:");
        if let Some(config) = existing_config {
            println!("  Embedding: {}", config.embedding_model.name);
            println!("  Reranker:  {}", config.reranker_model.name);
        }
        println!();
    }

    // Show options
    let default_embed = EmbeddingModelConfig::default();
    let default_rerank = RerankerModelConfig::default();
    println!("[D] Default - {} ({}MB) + {} ({}MB)",
        default_embed.name,
        default_embed.size_mb,
        default_rerank.name,
        default_rerank.size_mb
    );
    println!("[C] Custom  - Choose your models");
    println!();

    print!("Choice [D/c]: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    let config = if input == "c" || input == "custom" {
        run_custom_selection(existing_config)?
    } else {
        Config::default()
    };

    // Check if embedding model changed (requires reindex)
    let needs_reindex = if let Some(existing) = existing_config {
        existing.embedding_model != config.embedding_model
    } else {
        false
    };

    if needs_reindex {
        println!();
        println!("\x1b[33m⚠\x1b[0m  Embedding model changed. This requires reindexing.");
        println!("    All documents will be re-chunked and re-embedded.");
        println!();
        print!("Continue? [y/N]: ");
        io::stdout().flush()?;

        let mut confirm = String::new();
        io::stdin().read_line(&mut confirm)?;
        let confirm = confirm.trim().to_lowercase();

        if confirm != "y" && confirm != "yes" {
            return Ok(InitResult::Cancelled);
        }
    }

    // Download models
    println!("\n  Downloading Models (this may take a while for the first run)\n");
    let downloader = crate::setup::ModelDownloader::new();

    // 1. Embedding Model
    download_model(&downloader, &config.embedding_model).await?;

    // 2. Reranker Model
    download_model(&downloader, &config.reranker_model).await?;

    // 3. LLM (if Local)
    if let crate::config::LLMProviderType::Local = config.llm.provider {
        let phi3 = crate::setup::Phi3Model::new();
        download_model(&downloader, &phi3).await?;
    }

    // Save config
    config.save()?;

    Ok(InitResult::Configured(config))
}

/// Helper to download a model with simple progress
async fn download_model<M: crate::setup::ModelInfo>(
    downloader: &crate::setup::ModelDownloader,
    model: &M
) -> Result<()> {
    use std::io::Write;
    
    // FETCH METADATA (Async)
    let task = downloader.create_tasks(model).await?;
    let model_dir = downloader.model_cache_dir(&task.repo_id);
    
    println!("  {} ({} MB)", task.name, task.size_mb);
    
    let mut task_clone = task; 
    
    for file in &mut task_clone.files {
        if file.done {
             println!("    - {} (cached)", file.name);
             continue;
        }
        
        print!("    - {}... ", file.name);
        io::stdout().flush()?;
        
        // DOWNLOAD FILE (Async)
        downloader.download_file(
            file, 
            &model_dir, 
            task_clone.commit_hash.as_deref(),
            |_| {} // No callback for simple
        ).await?;
        println!("done");
    }
    println!();
    Ok(())
}

/// Run custom model selection
fn run_custom_selection(existing_config: Option<&Config>) -> Result<Config> {
    let embedding_model = select_embedding_model(existing_config)?;
    let reranker_model = select_reranker_model(existing_config)?;
    let llm_config = select_llm_provider(existing_config)?;

    Ok(Config {
        embedding_model,
        reranker_model,
        device: DevicePreference::default(),
        version: 2,
        llm: llm_config,
    })
}

/// Select LLM provider interactively
fn select_llm_provider(existing_config: Option<&Config>) -> Result<crate::config::LLMConfig> {
    use crate::config::{LLMConfig, LLMProviderType};

    println!();
    println!("LLM Provider (The Brain):");
    println!("  [1] OpenAI (Cloud) - Requires API Key");
    println!("  [2] Local (Phi-3)  - Offline, Private (~2GB download)");
    println!("  [3] Anthropic      - Cloud, Requires API Key"); 
    println!("  [4] None           - Skip for now");
    
    let current_provider = existing_config.map(|c| &c.llm.provider);
    let default = match current_provider {
        Some(LLMProviderType::Local) => 2,
        Some(LLMProviderType::Anthropic) => 3,
        Some(LLMProviderType::Disabled) => 4,
        _ => 1,
    };

    print!("Choice [{}]: ", default);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    let choice = if input.is_empty() {
        default
    } else {
        input.parse::<usize>().unwrap_or(default)
    };

    let (provider, needs_key) = match choice {
        2 => (LLMProviderType::Local, false),
        3 => (LLMProviderType::Anthropic, true),
        4 => (LLMProviderType::Disabled, false),
        _ => (LLMProviderType::OpenAI, true),
    };

    let mut api_key = existing_config.and_then(|c| c.llm.api_key.clone());

    if needs_key {
        println!();
        if let Some(ref k) = api_key {
            println!("Current API Key: {}...{}", &k[..4.min(k.len())], &k[k.len().saturating_sub(4)..]);
            print!("Enter new key (or press Enter to keep): ");
        } else {
            print!("Enter API Key (or press Enter to skip): ");
        }
        io::stdout().flush()?;

        let mut key_input = String::new();
        io::stdin().read_line(&mut key_input)?;
        let key_input = key_input.trim().to_string();

        if !key_input.is_empty() {
            api_key = Some(key_input);
        }
    } else if provider == LLMProviderType::Local {
        println!();
        println!("\x1b[33mNote: The first run will automatically download the Phi-3 model (~2GB).\x1b[0m");
    }

    Ok(LLMConfig {
        provider,
        model: None, // Use default for provider
        api_key,
    })
}

/// Select embedding model interactively
fn select_embedding_model(existing_config: Option<&Config>) -> Result<EmbeddingModelConfig> {
    println!();
    println!("Embedding model:");

    let models = EmbeddingModelConfig::curated_models();
    let current_id = existing_config.map(|c| &c.embedding_model.id);

    for (i, model) in models.iter().enumerate() {
        let current_marker = if Some(&model.id) == current_id { " ← current" } else { "" };
        println!("  [{}] {} ({}MB, {} dims){}",
            i + 1,
            model.name,
            model.size_mb,
            model.dimensions,
            current_marker
        );
    }
    println!();

    let default_idx = current_id
        .and_then(|id| models.iter().position(|m| &m.id == id))
        .unwrap_or(0);

    print!("Choice [{}]: ", default_idx + 1);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() {
        return Ok(models[default_idx].clone());
    }

    match input.parse::<usize>() {
        Ok(n) if n >= 1 && n <= models.len() => Ok(models[n - 1].clone()),
        _ => {
            println!("Invalid selection, using default.");
            Ok(models[default_idx].clone())
        }
    }
}

/// Select reranker model interactively
fn select_reranker_model(existing_config: Option<&Config>) -> Result<RerankerModelConfig> {
    println!();
    println!("Reranker model:");

    let models = RerankerModelConfig::curated_models();
    let current_id = existing_config.map(|c| &c.reranker_model.id);

    for (i, model) in models.iter().enumerate() {
        let current_marker = if Some(&model.id) == current_id { " ← current" } else { "" };
        println!("  [{}] {} ({}MB){}",
            i + 1,
            model.name,
            model.size_mb,
            current_marker
        );
    }
    println!();

    let default_idx = current_id
        .and_then(|id| models.iter().position(|m| &m.id == id))
        .unwrap_or(0);

    print!("Choice [{}]: ", default_idx + 1);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() {
        return Ok(models[default_idx].clone());
    }

    match input.parse::<usize>() {
        Ok(n) if n >= 1 && n <= models.len() => Ok(models[n - 1].clone()),
        _ => {
            println!("Invalid selection, using default.");
            Ok(models[default_idx].clone())
        }
    }
}

/// Display status information
pub fn show_status(config: &Config, sources: usize, documents: usize, chunks: usize) {
    println!("Eywa v{} - The memory your team never loses\n",
        env!("CARGO_PKG_VERSION")
    );

    println!("Status:");
    println!("  Sources:   {}", sources);
    println!("  Documents: {}", documents);
    println!("  Chunks:    {}", chunks);
    println!();

    println!("Models:");
    println!("  Embedding: {}", config.embedding_model.name);
    println!("  Reranker:  {}", config.reranker_model.name);
    println!();

    println!("Run 'eywa --help' for commands.");
}

/// Display first-run welcome message
pub fn show_welcome() {
    println!("Eywa v{} - The memory your team never loses\n",
        env!("CARGO_PKG_VERSION")
    );
    println!("First run detected. Let's set you up.\n");
}

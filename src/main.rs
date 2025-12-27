//! Eywa CLI
//!
//! Personal knowledge base with local embeddings.
//!
//! Commands:
//!   ingest  - Ingest documents from a file or directory
//!   search  - Search for similar documents
//!   sources - List all sources
//!   docs    - List documents in a source
//!   delete  - Delete a source
//!   reset   - Reset config and data (keeps models)
//!   hard-reset - Delete everything including models
//!   uninstall - Full uninstall with instructions
//!   serve   - Start HTTP server
//!   mcp     - Start MCP server (for Claude/Cursor)
//!   info    - Show model info
//!   storage - Show storage usage
//!   init    - Configure models

mod commands;
mod server;
mod mcp;
mod utils;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use eywa::{
    db, run_download_wizard, run_init, show_status, show_welcome,
    Config, Embedder, InitResult, Reranker, VectorDB,
};
use utils::expand_path;

#[derive(Parser)]
#[command(name = "eywa")]
#[command(about = "Personal knowledge base with local embeddings")]
#[command(version)]
struct Cli {
    /// Data directory for storing the database
    #[arg(short, long, default_value = "~/.eywa/data")]
    data_dir: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest documents from a file or directory
    Ingest {
        /// Source ID (name for this collection)
        #[arg(short, long)]
        source: String,

        /// Path to file or directory to ingest
        path: PathBuf,
    },

    /// Search for documents
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Filter by source ID
        #[arg(short, long)]
        source: Option<String>,
    },

    /// List all sources
    Sources,

    /// List documents in a source
    Docs {
        /// Source ID
        source: String,
    },

    /// Delete a source
    Delete {
        /// Source ID to delete
        source: String,
    },

    /// Reset - delete ~/.eywa (config, data, sqlite). Keeps models.
    Reset,

    /// Hard reset - delete everything including downloaded models
    HardReset,

    /// Uninstall - delete all data and show binary removal instructions
    Uninstall,

    /// Start HTTP server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8005")]
        port: u16,
    },

    /// Start MCP server (for Claude/Cursor)
    Mcp,

    /// Show model info
    Info,

    /// Show storage usage (data, models, total)
    Storage,

    /// Run initialization flow (re-configure models)
    Init {
        /// Use default models without prompts (for CI/scripting)
        #[arg(long)]
        default: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let data_dir = expand_path(&cli.data_dir);

    // Ensure data directory exists
    std::fs::create_dir_all(&data_dir)?;

    match cli.command {
        None => {
            // No command = show status or run init if first run
            match Config::load()? {
                Some(config) => {
                    // Show status
                    let db = VectorDB::new(&data_dir).await?;
                    let sources = db.list_sources().await?;
                    let total_chunks: usize = sources.iter().map(|s| s.chunk_count as usize).sum();
                    let total_docs: usize = {
                        let mut count = 0;
                        for source in &sources {
                            count += db.list_documents(&source.name, Some(db::MAX_QUERY_LIMIT)).await?.len();
                        }
                        count
                    };
                    show_status(&config, sources.len(), total_docs, total_chunks);
                }
                None => {
                    // First run - run init
                    show_welcome();
                    match run_init(None)? {
                        InitResult::Configured(config) => {
                            println!("\n\x1b[32m✓\x1b[0m Configuration saved!\n");

                            // Run the TUI download wizard
                            run_download_wizard(&config)?;

                            // Verify models load correctly (uses hf_hub cache)
                            let _embedder = Embedder::new()?;
                            let _reranker = Reranker::new()?;

                            println!("\n\x1b[32m✓\x1b[0m Setup complete! Run 'eywa --help' to get started.");
                        }
                        InitResult::Cancelled => {
                            println!("\nSetup cancelled.");
                        }
                    }
                }
            }
        }

        Some(Commands::Ingest { source, path }) => {
            commands::run_ingest(&data_dir, &source, &path).await?;
        }

        Some(Commands::Search { query, limit, source: _ }) => {
            commands::run_search(&data_dir, &query, limit).await?;
        }

        Some(Commands::Sources) => {
            commands::run_sources(&data_dir).await?;
        }

        Some(Commands::Docs { source }) => {
            commands::run_docs(&data_dir, &source).await?;
        }

        Some(Commands::Delete { source }) => {
            commands::run_delete(&data_dir, &source).await?;
        }

        Some(Commands::Reset) => {
            commands::run_reset()?;
        }

        Some(Commands::HardReset) => {
            commands::run_hard_reset()?;
        }

        Some(Commands::Uninstall) => {
            commands::run_uninstall()?;
        }

        Some(Commands::Serve { port }) => {
            println!("Starting server on http://localhost:{}...", port);
            server::run_server(&data_dir, port).await?;
        }

        Some(Commands::Mcp) => {
            mcp::run_mcp_server(&data_dir).await?;
        }

        Some(Commands::Info) => {
            commands::run_info(&data_dir)?;
        }

        Some(Commands::Storage) => {
            commands::run_storage(&data_dir)?;
        }

        Some(Commands::Init { default }) => {
            commands::run_init_command(&data_dir, default).await?;
        }
    }

    Ok(())
}

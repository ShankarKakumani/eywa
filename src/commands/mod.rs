//! CLI command handlers

pub mod ingest;
pub mod search;
pub mod sources;
pub mod reset;
pub mod info;
pub mod init;

pub use ingest::run_ingest;
pub use search::run_search;
pub use sources::{run_sources, run_docs, run_delete};
pub use reset::{run_reset, run_hard_reset, run_uninstall};
pub use info::{run_info, run_storage};
pub use init::run_init_command;

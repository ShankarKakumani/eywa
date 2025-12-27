//! Reset, hard reset, and uninstall command handlers

use anyhow::Result;
use std::io::Write;

pub fn run_reset() -> Result<()> {
    let eywa_dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?
        .join(".eywa");

    if eywa_dir.exists() {
        std::fs::remove_dir_all(&eywa_dir)?;
        println!("\x1b[32m✓\x1b[0m Deleted ~/.eywa/");
        println!("\nRun 'eywa' to set up again.");
    } else {
        println!("Nothing to reset - ~/.eywa/ does not exist.");
    }

    Ok(())
}

pub fn run_hard_reset() -> Result<()> {
    // Get paths
    let home = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
    let eywa_dir = home.join(".eywa");
    let hf_cache = home.join(".cache").join("huggingface").join("hub");
    let fastembed_cache = home.join(".fastembed_cache");

    // Show what will be deleted
    println!("\n\x1b[1;31m⚠ HARD RESET\x1b[0m\n");
    println!("This will permanently delete:");
    println!("  • \x1b[33m~/.eywa/\x1b[0m (config, data, content database)");
    println!("  • \x1b[33m~/.cache/huggingface/hub/\x1b[0m (models)");
    println!("  • \x1b[33m~/.fastembed_cache/\x1b[0m (legacy models)");
    println!();

    // Confirmation prompt
    print!("Type '\x1b[1myes\x1b[0m' to confirm: ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim() != "yes" {
        println!("\nAborted. No data was deleted.");
        return Ok(());
    }

    // Delete eywa directory
    if eywa_dir.exists() {
        std::fs::remove_dir_all(&eywa_dir)?;
        println!("\n\x1b[32m✓\x1b[0m Deleted ~/.eywa/");
    } else {
        println!("\n\x1b[90m~/.eywa/ does not exist\x1b[0m");
    }

    // Delete HuggingFace cache
    if hf_cache.exists() {
        std::fs::remove_dir_all(&hf_cache)?;
        println!("\x1b[32m✓\x1b[0m Deleted ~/.cache/huggingface/hub/");
    } else {
        println!("\x1b[90m~/.cache/huggingface/hub/ does not exist\x1b[0m");
    }

    // Delete legacy fastembed cache
    if fastembed_cache.exists() {
        std::fs::remove_dir_all(&fastembed_cache)?;
        println!("\x1b[32m✓\x1b[0m Deleted ~/.fastembed_cache/");
    }

    println!("\n\x1b[32mHard reset complete.\x1b[0m Run 'eywa' to set up again.");

    Ok(())
}

pub fn run_uninstall() -> Result<()> {
    // Get paths
    let home = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
    let eywa_dir = home.join(".eywa");
    let hf_cache = home.join(".cache").join("huggingface").join("hub");
    let fastembed_cache = home.join(".fastembed_cache");

    // Show what will be deleted
    println!("\n\x1b[1;31m⚠ UNINSTALL EYWA\x1b[0m\n");
    println!("This will permanently delete:");
    println!("  • \x1b[33m~/.eywa/\x1b[0m (config, data, content database)");
    println!("  • \x1b[33m~/.cache/huggingface/hub/\x1b[0m (models)");
    println!("  • \x1b[33m~/.fastembed_cache/\x1b[0m (legacy models)");
    println!();

    // Confirmation prompt
    print!("Type '\x1b[1myes\x1b[0m' to confirm: ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim() != "yes" {
        println!("\nAborted. Nothing was deleted.");
        return Ok(());
    }

    // Delete eywa directory
    if eywa_dir.exists() {
        std::fs::remove_dir_all(&eywa_dir)?;
        println!("\n\x1b[32m✓\x1b[0m Deleted ~/.eywa/");
    } else {
        println!("\n\x1b[90m~/.eywa/ does not exist\x1b[0m");
    }

    // Delete HuggingFace cache
    if hf_cache.exists() {
        std::fs::remove_dir_all(&hf_cache)?;
        println!("\x1b[32m✓\x1b[0m Deleted ~/.cache/huggingface/hub/");
    } else {
        println!("\x1b[90m~/.cache/huggingface/hub/ does not exist\x1b[0m");
    }

    // Delete legacy fastembed cache
    if fastembed_cache.exists() {
        std::fs::remove_dir_all(&fastembed_cache)?;
        println!("\x1b[32m✓\x1b[0m Deleted ~/.fastembed_cache/");
    }

    // Show binary removal instructions
    println!("\n\x1b[32mData deleted.\x1b[0m To complete uninstallation, remove the binary:\n");
    println!("  \x1b[36mHomebrew:\x1b[0m  brew uninstall eywa");
    println!("  \x1b[36mCargo:\x1b[0m     cargo uninstall eywa");
    println!("  \x1b[36mManual:\x1b[0m    rm $(which eywa)");

    Ok(())
}

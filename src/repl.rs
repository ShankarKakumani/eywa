//! Interactive REPL mode for Eywa
//!
//! Run `eywa` with no arguments to enter interactive mode.
//! Features Claude Code-style inline rendering with dropdown below input.

use anyhow::Result;
use colored::*;
use crossterm::{
    cursor::{MoveLeft, RestorePosition, SavePosition},
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
};
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db;
use crate::{ContentStore, Embedder, Ingester, SearchEngine, SearchResult, VectorDB};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Example queries to show in banner
const EXAMPLES: &[(&str, &str)] = &[
    ("how does authentication work?", "Search your docs"),
    ("what's the API rate limit?", "Find specific info"),
    ("/ingest ~/docs --source notes", "Add documents"),
    ("/sources", "List all sources"),
    ("explain the login flow", "Natural language search"),
    ("where is the config file?", "Find locations"),
    ("/help", "See all commands"),
    ("summarize the readme", "Query your knowledge"),
    ("what frameworks are used?", "Explore codebase"),
    ("/info", "System information"),
];

/// Get two random examples
fn get_random_examples() -> [(&'static str, &'static str); 2] {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as usize;

    let idx1 = seed % EXAMPLES.len();
    let idx2 = (seed / 7 + 3) % EXAMPLES.len();
    // Ensure different examples
    let idx2 = if idx2 == idx1 { (idx2 + 1) % EXAMPLES.len() } else { idx2 };

    [EXAMPLES[idx1], EXAMPLES[idx2]]
}

/// Command definition with name and description
struct Command {
    name: &'static str,
    description: &'static str,
}

const COMMANDS: &[Command] = &[
    Command { name: "/add", description: "Add a new document" },
    Command { name: "/ingest", description: "Ingest files from path" },
    Command { name: "/sources", description: "List all sources" },
    Command { name: "/docs", description: "List documents in source" },
    Command { name: "/delete", description: "Delete source or document" },
    Command { name: "/info", description: "Show system info" },
    Command { name: "/clear", description: "Clear screen" },
    Command { name: "/help", description: "Show this help" },
    Command { name: "/exit", description: "Exit" },
];

/// Print the welcome banner
fn print_banner(doc_count: u64) {
    let doc_text = if doc_count == 0 {
        "No documents".yellow().to_string()
    } else if doc_count == 1 {
        "1 document".green().to_string()
    } else {
        format!("{} documents", doc_count).green().to_string()
    };

    let examples = get_random_examples();

    println!();
    println!("      {}  ·  {}", "✧".cyan(), "✧".cyan());
    println!("       {}   {}", "\\".cyan(), "/".cyan());
    println!(
        "     {}        {} v{}",
        ".--'•'--.".cyan(),
        "Eywa".green().bold(),
        VERSION
    );
    println!(
        "    {}       Personal Knowledge Base",
        "( ◠  ‿  ◠ )".cyan()
    );
    println!("     {}", "\\       /".cyan());
    println!("      {}         {} indexed", "'-----'".cyan(), doc_text);
    println!("    ·    {}    ·", "✧".cyan());
    println!();
    println!("  {}", "Try:".dimmed());
    println!(
        "    {} {}  {}",
        ">".green(),
        examples[0].0.white(),
        format!("({})", examples[0].1).dimmed()
    );
    println!(
        "    {} {}  {}",
        ">".green(),
        examples[1].0.white(),
        format!("({})", examples[1].1).dimmed()
    );
    println!();
}

/// Filter commands based on input
fn filter_commands(input: &str) -> Vec<usize> {
    if !input.starts_with('/') {
        return vec![];
    }
    let filter = input.to_lowercase();
    COMMANDS
        .iter()
        .enumerate()
        .filter(|(_, cmd)| cmd.name.to_lowercase().starts_with(&filter))
        .map(|(i, _)| i)
        .collect()
}

/// Render dropdown below current line
fn render_dropdown(
    stdout: &mut io::Stdout,
    filtered: &[usize],
    selected: usize,
) -> Result<()> {
    for (i, &cmd_idx) in filtered.iter().enumerate() {
        let cmd = &COMMANDS[cmd_idx];
        if i == selected {
            // Highlighted row
            println!(
                "\r  {}",
                format!("{:<12} {}", cmd.name, cmd.description)
                    .on_bright_black()
                    .white()
            );
        } else {
            println!(
                "\r  {}  {}",
                cmd.name.white(),
                cmd.description.dimmed()
            );
        }
    }
    stdout.flush()?;
    Ok(())
}

/// Clear from saved position down (restore + clear)
fn clear_from_saved(stdout: &mut io::Stdout) -> Result<()> {
    execute!(stdout, RestorePosition, Clear(ClearType::FromCursorDown))?;
    Ok(())
}

/// Redraw the input line at current position
fn redraw_input(stdout: &mut io::Stdout, input: &str) -> Result<()> {
    print!("{} {}", ">".green().bold(), input);
    stdout.flush()?;
    Ok(())
}

/// Redraw input and position cursor correctly
fn redraw_input_with_cursor(stdout: &mut io::Stdout, input: &str, cursor_pos: usize) -> Result<()> {
    // For multi-line input, use the multiline renderer
    if input.contains('\n') {
        return redraw_input_multiline(stdout, input, cursor_pos);
    }
    print!("{} {}", ">".green().bold(), input);
    stdout.flush()?;
    // Move cursor back to correct position if not at end
    let chars_from_end = input.chars().count() - cursor_pos;
    if chars_from_end > 0 {
        execute!(stdout, MoveLeft(chars_from_end as u16))?;
    }
    Ok(())
}

/// Redraw multi-line input with proper cursor positioning
fn redraw_input_multiline(stdout: &mut io::Stdout, input: &str, cursor_pos: usize) -> Result<()> {
    use crossterm::cursor::MoveUp;

    let lines: Vec<&str> = input.split('\n').collect();

    // Print each line with continuation prompt
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            print!("{} {}", ">".green().bold(), line);
        } else {
            print!("\n{} {}", ".".cyan(), line);
        }
    }
    stdout.flush()?;

    // Calculate cursor position
    // Find which line and column the cursor is on
    let chars: Vec<char> = input.chars().collect();
    let mut line_num = 0;
    let mut col = 0;
    for (i, c) in chars.iter().enumerate() {
        if i == cursor_pos {
            break;
        }
        if *c == '\n' {
            line_num += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    // Move cursor to correct position
    let total_lines = lines.len();
    let lines_up = total_lines - 1 - line_num;
    if lines_up > 0 {
        execute!(stdout, MoveUp(lines_up as u16))?;
    }

    // Move to correct column (account for prompt "> " or ". ")
    let current_line_len = lines[line_num].chars().count();
    let chars_from_end = current_line_len - col;
    if chars_from_end > 0 {
        execute!(stdout, MoveLeft(chars_from_end as u16))?;
    }

    Ok(())
}

/// Save current cursor position
fn save_position(stdout: &mut io::Stdout) -> Result<()> {
    execute!(stdout, SavePosition)?;
    Ok(())
}

/// Run the interactive REPL
pub async fn run_repl(data_dir: &str) -> Result<()> {
    // Initialize components (downloads models on first run)
    let embedder = Embedder::new()?;
    let mut db = VectorDB::new(data_dir).await?;
    let content_store = ContentStore::open(&std::path::Path::new(data_dir).join("content.db"))?;
    let search_engine = SearchEngine::with_reranker()?;

    // Get stats for banner
    let sources = db.list_sources().await?;
    let doc_count: u64 = sources.iter().map(|s| s.chunk_count).sum();

    // Print banner once
    print_banner(doc_count);

    let mut stdout = io::stdout();

    loop {
        // Read input with dropdown support (handles prompt internally)
        let input = read_input_with_dropdown(&mut stdout).await?;

        if input.is_empty() {
            continue;
        }

        // Handle input
        if input.starts_with('/') {
            let should_exit = handle_command(&input, &embedder, &mut db, &search_engine, data_dir).await?;
            if should_exit {
                println!("{}", "Goodbye!".cyan());
                break;
            }
        } else {
            // Search
            do_search(&input, &embedder, &db, &content_store, &search_engine).await?;
        }

        println!(); // Empty line after output
    }

    Ok(())
}

/// Find word boundary going left from position
fn find_word_left(input: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let chars: Vec<char> = input.chars().collect();
    let mut i = pos - 1;
    // Skip whitespace
    while i > 0 && chars[i].is_whitespace() {
        i -= 1;
    }
    // Skip word characters
    while i > 0 && !chars[i - 1].is_whitespace() {
        i -= 1;
    }
    i
}

/// Find word boundary going right from position
fn find_word_right(input: &str, pos: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    if pos >= len {
        return len;
    }
    let mut i = pos;
    // Skip current word
    while i < len && !chars[i].is_whitespace() {
        i += 1;
    }
    // Skip whitespace
    while i < len && chars[i].is_whitespace() {
        i += 1;
    }
    i
}

/// Read input with live dropdown filtering
async fn read_input_with_dropdown(stdout: &mut io::Stdout) -> Result<String> {
    let mut input = String::new();
    let mut cursor_pos: usize = 0;
    let mut selected: usize = 0;
    let mut has_dropdown = false;
    let mut last_was_esc = false; // Track ESC for macOS Option+Arrow sequences

    // Save position at start of line, then show prompt
    save_position(stdout)?;
    redraw_input_with_cursor(stdout, &input, cursor_pos)?;

    enable_raw_mode()?;

    loop {
        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                // If ESC was pressed alone (not followed by b/f/backspace), treat as clear
                if last_was_esc && !matches!(key.code, KeyCode::Char('b') | KeyCode::Char('f') | KeyCode::Backspace) {
                    last_was_esc = false;
                    // Standalone ESC - clear input
                    input.clear();
                    cursor_pos = 0;
                    selected = 0;
                    has_dropdown = false;
                    clear_from_saved(stdout)?;
                    redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                    // Don't process this key as ESC was meant to clear
                    // But we still need to handle the current key if it's not ESC
                    if matches!(key.code, KeyCode::Esc) {
                        continue;
                    }
                }

                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        clear_from_saved(stdout)?;
                        disable_raw_mode()?;
                        println!();
                        return Ok("/exit".to_string());
                    }
                    KeyCode::Esc => {
                        // Could be start of ESC sequence (Option+Arrow on macOS)
                        // Wait briefly to see if more keys follow
                        last_was_esc = true;
                        continue; // Don't clear yet, wait for next key
                    }
                    // Option+Enter: insert newline (multi-line input)
                    KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => {
                        // Insert newline at cursor
                        let chars: Vec<char> = input.chars().collect();
                        input = chars[..cursor_pos]
                            .iter()
                            .chain(std::iter::once(&'\n'))
                            .chain(chars[cursor_pos..].iter())
                            .collect();
                        cursor_pos += 1;
                        has_dropdown = false;
                        clear_from_saved(stdout)?;
                        redraw_input_multiline(stdout, &input, cursor_pos)?;
                    }
                    KeyCode::Enter => {
                        let filtered = filter_commands(&input);
                        if !filtered.is_empty() && has_dropdown {
                            // Select command from dropdown
                            input = COMMANDS[filtered[selected]].name.to_string();
                            cursor_pos = input.chars().count();
                            has_dropdown = false;
                            selected = 0;
                            // Redraw
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                        } else {
                            // Submit input
                            clear_from_saved(stdout)?;
                            redraw_input(stdout, &input)?;
                            disable_raw_mode()?;
                            println!();
                            return Ok(input);
                        }
                    }
                    KeyCode::Tab => {
                        let filtered = filter_commands(&input);
                        if !filtered.is_empty() && has_dropdown {
                            input = COMMANDS[filtered[selected]].name.to_string();
                            cursor_pos = input.chars().count();
                            selected = 0;
                            // Redraw with new input
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            // Check if still has dropdown
                            let new_filtered = filter_commands(&input);
                            if !new_filtered.is_empty() {
                                println!();
                                render_dropdown(stdout, &new_filtered, selected)?;
                                has_dropdown = true;
                            } else {
                                has_dropdown = false;
                            }
                        }
                    }
                    // Word movement: Option+Left (Alt+Left or ESC+b on macOS)
                    KeyCode::Left if key.modifiers.contains(KeyModifiers::ALT) => {
                        let new_pos = find_word_left(&input, cursor_pos);
                        if new_pos != cursor_pos {
                            cursor_pos = new_pos;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    // Word movement: Option+Right (Alt+Right or ESC+f on macOS)
                    KeyCode::Right if key.modifiers.contains(KeyModifiers::ALT) => {
                        let new_pos = find_word_right(&input, cursor_pos);
                        if new_pos != cursor_pos {
                            cursor_pos = new_pos;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    // macOS: Option+Left sends ESC+b (backward word)
                    KeyCode::Char('b') if last_was_esc || key.modifiers.contains(KeyModifiers::ALT) => {
                        last_was_esc = false;
                        let new_pos = find_word_left(&input, cursor_pos);
                        if new_pos != cursor_pos {
                            cursor_pos = new_pos;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    // macOS: Option+Right sends ESC+f (forward word)
                    KeyCode::Char('f') if last_was_esc || key.modifiers.contains(KeyModifiers::ALT) => {
                        last_was_esc = false;
                        let new_pos = find_word_right(&input, cursor_pos);
                        if new_pos != cursor_pos {
                            cursor_pos = new_pos;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    // Ctrl+W or Option+Backspace: delete word backward
                    KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if cursor_pos > 0 {
                            let new_pos = find_word_left(&input, cursor_pos);
                            let chars: Vec<char> = input.chars().collect();
                            input = chars[..new_pos]
                                .iter()
                                .chain(chars[cursor_pos..].iter())
                                .collect();
                            cursor_pos = new_pos;
                            selected = 0;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            let filtered = filter_commands(&input);
                            if !filtered.is_empty() {
                                println!();
                                render_dropdown(stdout, &filtered, selected)?;
                                has_dropdown = true;
                            } else {
                                has_dropdown = false;
                            }
                        }
                    }
                    // macOS: Option+Delete sends ESC+Backspace - delete word backward
                    KeyCode::Backspace if last_was_esc || key.modifiers.contains(KeyModifiers::ALT) => {
                        last_was_esc = false;
                        if cursor_pos > 0 {
                            let new_pos = find_word_left(&input, cursor_pos);
                            let chars: Vec<char> = input.chars().collect();
                            input = chars[..new_pos]
                                .iter()
                                .chain(chars[cursor_pos..].iter())
                                .collect();
                            cursor_pos = new_pos;
                            selected = 0;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            let filtered = filter_commands(&input);
                            if !filtered.is_empty() {
                                println!();
                                render_dropdown(stdout, &filtered, selected)?;
                                has_dropdown = true;
                            } else {
                                has_dropdown = false;
                            }
                        }
                    }
                    // Option+Fn+Delete or Ctrl+Delete: delete word forward
                    KeyCode::Delete if key.modifiers.contains(KeyModifiers::ALT) || key.modifiers.contains(KeyModifiers::CONTROL) => {
                        let len = input.chars().count();
                        if cursor_pos < len {
                            let new_pos = find_word_right(&input, cursor_pos);
                            let chars: Vec<char> = input.chars().collect();
                            input = chars[..cursor_pos]
                                .iter()
                                .chain(chars[new_pos..].iter())
                                .collect();
                            selected = 0;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            let filtered = filter_commands(&input);
                            if !filtered.is_empty() {
                                println!();
                                render_dropdown(stdout, &filtered, selected)?;
                                has_dropdown = true;
                            } else {
                                has_dropdown = false;
                            }
                        }
                    }
                    // Ctrl+U: delete from cursor to start of line
                    KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if cursor_pos > 0 {
                            let chars: Vec<char> = input.chars().collect();
                            input = chars[cursor_pos..].iter().collect();
                            cursor_pos = 0;
                            selected = 0;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            let filtered = filter_commands(&input);
                            if !filtered.is_empty() {
                                println!();
                                render_dropdown(stdout, &filtered, selected)?;
                                has_dropdown = true;
                            } else {
                                has_dropdown = false;
                            }
                        }
                    }
                    // Ctrl+K: delete from cursor to end of line
                    KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        let len = input.chars().count();
                        if cursor_pos < len {
                            let chars: Vec<char> = input.chars().collect();
                            input = chars[..cursor_pos].iter().collect();
                            selected = 0;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            let filtered = filter_commands(&input);
                            if !filtered.is_empty() {
                                println!();
                                render_dropdown(stdout, &filtered, selected)?;
                                has_dropdown = true;
                            } else {
                                has_dropdown = false;
                            }
                        }
                    }
                    // Ctrl+L: clear screen, keep input
                    KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Clear screen
                        print!("\x1B[2J\x1B[1;1H");
                        stdout.flush()?;
                        // Redraw prompt and input
                        save_position(stdout)?;
                        redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                        if has_dropdown {
                            let filtered = filter_commands(&input);
                            if !filtered.is_empty() {
                                println!();
                                render_dropdown(stdout, &filtered, selected)?;
                            }
                        }
                    }
                    // Ctrl+D: exit on empty line, or delete char at cursor
                    KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if input.is_empty() {
                            // Exit on empty line
                            clear_from_saved(stdout)?;
                            disable_raw_mode()?;
                            println!();
                            return Ok("/exit".to_string());
                        } else {
                            // Delete char at cursor (same as Delete key)
                            let len = input.chars().count();
                            if cursor_pos < len {
                                let chars: Vec<char> = input.chars().collect();
                                input = chars[..cursor_pos]
                                    .iter()
                                    .chain(chars[cursor_pos + 1..].iter())
                                    .collect();
                                selected = 0;
                                clear_from_saved(stdout)?;
                                redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                    has_dropdown = true;
                                } else {
                                    has_dropdown = false;
                                }
                            }
                        }
                    }
                    // Ctrl+T: transpose characters
                    KeyCode::Char('t') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        let len = input.chars().count();
                        if cursor_pos > 0 && len >= 2 {
                            let mut chars: Vec<char> = input.chars().collect();
                            let swap_pos = if cursor_pos == len { cursor_pos - 1 } else { cursor_pos };
                            if swap_pos > 0 {
                                chars.swap(swap_pos - 1, swap_pos);
                                input = chars.into_iter().collect();
                                // Move cursor forward if not at end
                                if cursor_pos < len {
                                    cursor_pos += 1;
                                }
                                clear_from_saved(stdout)?;
                                redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                                if has_dropdown {
                                    let filtered = filter_commands(&input);
                                    if !filtered.is_empty() {
                                        println!();
                                        render_dropdown(stdout, &filtered, selected)?;
                                    }
                                }
                            }
                        }
                    }
                    // Ctrl+A: go to start of line
                    KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if cursor_pos > 0 {
                            cursor_pos = 0;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    // Ctrl+E: go to end of line
                    KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        let len = input.chars().count();
                        if cursor_pos < len {
                            cursor_pos = len;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    // Character movement: Left arrow
                    KeyCode::Left => {
                        if cursor_pos > 0 {
                            cursor_pos -= 1;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    // Character movement: Right arrow
                    KeyCode::Right => {
                        if cursor_pos < input.chars().count() {
                            cursor_pos += 1;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    KeyCode::Up => {
                        let filtered = filter_commands(&input);
                        if !filtered.is_empty() && has_dropdown && selected > 0 {
                            selected -= 1;
                            // Redraw dropdown only
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            println!();
                            render_dropdown(stdout, &filtered, selected)?;
                        }
                    }
                    KeyCode::Down => {
                        let filtered = filter_commands(&input);
                        if !filtered.is_empty() && has_dropdown && selected < filtered.len() - 1 {
                            selected += 1;
                            // Redraw dropdown only
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            println!();
                            render_dropdown(stdout, &filtered, selected)?;
                        }
                    }
                    // Home key - go to beginning
                    KeyCode::Home => {
                        if cursor_pos > 0 {
                            cursor_pos = 0;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    // End key - go to end
                    KeyCode::End => {
                        let len = input.chars().count();
                        if cursor_pos < len {
                            cursor_pos = len;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            if has_dropdown {
                                let filtered = filter_commands(&input);
                                if !filtered.is_empty() {
                                    println!();
                                    render_dropdown(stdout, &filtered, selected)?;
                                }
                            }
                        }
                    }
                    KeyCode::Backspace => {
                        if cursor_pos > 0 {
                            // Remove character before cursor
                            let chars: Vec<char> = input.chars().collect();
                            input = chars[..cursor_pos - 1]
                                .iter()
                                .chain(chars[cursor_pos..].iter())
                                .collect();
                            cursor_pos -= 1;
                            selected = 0;
                            // Redraw everything
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            let filtered = filter_commands(&input);
                            if !filtered.is_empty() {
                                println!();
                                render_dropdown(stdout, &filtered, selected)?;
                                has_dropdown = true;
                            } else {
                                has_dropdown = false;
                            }
                        }
                    }
                    // Delete key - delete character at cursor
                    KeyCode::Delete => {
                        let len = input.chars().count();
                        if cursor_pos < len {
                            let chars: Vec<char> = input.chars().collect();
                            input = chars[..cursor_pos]
                                .iter()
                                .chain(chars[cursor_pos + 1..].iter())
                                .collect();
                            selected = 0;
                            clear_from_saved(stdout)?;
                            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                            let filtered = filter_commands(&input);
                            if !filtered.is_empty() {
                                println!();
                                render_dropdown(stdout, &filtered, selected)?;
                                has_dropdown = true;
                            } else {
                                has_dropdown = false;
                            }
                        }
                    }
                    KeyCode::Char(c) => {
                        // Insert character at cursor position
                        let chars: Vec<char> = input.chars().collect();
                        input = chars[..cursor_pos]
                            .iter()
                            .chain(std::iter::once(&c))
                            .chain(chars[cursor_pos..].iter())
                            .collect();
                        cursor_pos += 1;
                        selected = 0;
                        // Redraw everything
                        clear_from_saved(stdout)?;
                        redraw_input_with_cursor(stdout, &input, cursor_pos)?;
                        let filtered = filter_commands(&input);
                        if !filtered.is_empty() {
                            println!();
                            render_dropdown(stdout, &filtered, selected)?;
                            has_dropdown = true;
                        } else {
                            has_dropdown = false;
                        }
                    }
                    _ => {
                        last_was_esc = false;
                    }
                }
            }
        } else if last_was_esc {
            // ESC was pressed but no follow-up key within timeout
            // Treat as standalone ESC - clear input
            last_was_esc = false;
            input.clear();
            cursor_pos = 0;
            selected = 0;
            has_dropdown = false;
            clear_from_saved(stdout)?;
            redraw_input_with_cursor(stdout, &input, cursor_pos)?;
        }
    }
}

/// Handle slash commands. Returns true if should exit.
async fn handle_command(
    input: &str,
    embedder: &Embedder,
    db: &mut VectorDB,
    _search_engine: &SearchEngine,
    data_dir: &str,
) -> Result<bool> {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0].to_lowercase();
    let args = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd.as_str() {
        "/exit" | "/quit" | "/q" => {
            return Ok(true);
        }
        "/help" | "/h" | "/?" => {
            println!("{}", "Commands:".green().bold());
            println!();
            println!("  {}           {}", "<query>".dimmed(), "Search for documents (default)".white());
            println!();
            for cmd in COMMANDS {
                println!("  {}  {}", format!("{:<12}", cmd.name).dimmed(), cmd.description.white());
            }
        }
        "/clear" => {
            print!("\x1B[2J\x1B[1;1H");
            io::stdout().flush()?;
        }
        "/sources" | "/s" => {
            let sources = db.list_sources().await?;
            if sources.is_empty() {
                println!("{}", "No sources found. Use /add to add documents.".yellow());
            } else {
                println!("{}", "Sources:".green().bold());
                println!();
                for source in sources {
                    println!(
                        "  {}  {} docs",
                        source.name.white().bold(),
                        source.chunk_count.to_string().cyan()
                    );
                }
            }
        }
        "/docs" | "/d" => {
            if args.is_empty() {
                println!("{}", "Usage: /docs <source>".yellow());
            } else {
                let docs = db.list_documents(args, Some(db::MAX_QUERY_LIMIT)).await?;
                if docs.is_empty() {
                    println!("{}", format!("No documents in source '{}'.", args).yellow());
                } else {
                    println!("{} '{}':", "Documents in".green().bold(), args);
                    println!();
                    for doc in docs {
                        println!(
                            "  {} - {} ({} chars)",
                            doc.id[..8].cyan(),
                            doc.title.white(),
                            doc.content_length.to_string().dimmed()
                        );
                    }
                }
            }
        }
        "/delete" | "/del" => {
            if args.is_empty() {
                println!("{}", "Usage: /delete <source-or-doc-id>".yellow());
            } else {
                db.delete_source(args).await?;
                println!("{} '{}'", "Deleted".green().bold(), args);
            }
        }
        "/info" => {
            println!("{}", "Eywa - Personal Knowledge Base".green().bold());
            println!();
            println!("  Version:    {}", VERSION.white());
            println!("  Model:      {}", "all-MiniLM-L6-v2 (Candle)".white());
            println!("  Dimensions: {}", "384".white());
            println!("  Database:   {}", "LanceDB".white());
            println!("  Data dir:   {}", data_dir.white());
        }
        "/add" | "/a" => {
            println!("{}", "Note: /add requires stdin input. Use CLI: eywa add".yellow());
        }
        "/ingest" | "/i" => {
            if args.is_empty() {
                println!("{}", "Usage: /ingest <path> [--source <name>]".yellow());
            } else {
                let parts: Vec<&str> = args.split("--source").collect();
                let path = parts[0].trim();
                let source = parts.get(1).map(|s| s.trim()).unwrap_or("default");

                println!("{} from {}...", "Ingesting".green().bold(), path);
                let ingester = Ingester::new(embedder);
                let data_path = std::path::Path::new(data_dir);
                match ingester.ingest_from_path(db, data_path, source, path).await {
                    Ok(result) => {
                        println!(
                            "{} {} documents ({} chunks) to '{}'",
                            "Added".green().bold(),
                            result.documents_created.to_string().white(),
                            result.chunks_created.to_string().yellow(),
                            source.cyan()
                        );
                    }
                    Err(e) => {
                        println!("{} {}", "Error:".red().bold(), e);
                    }
                }
            }
        }
        _ => {
            println!("{} Unknown command: {}", "Error:".red().bold(), cmd);
            println!("Type {} for available commands.", "/help".yellow());
        }
    }

    Ok(false)
}

/// Perform a search and display results
async fn do_search(
    query: &str,
    embedder: &Embedder,
    db: &VectorDB,
    content_store: &ContentStore,
    search_engine: &SearchEngine,
) -> Result<()> {
    let query_embedding = embedder.embed(query)?;
    // Get chunk metadata from LanceDB
    let chunk_metas = db.search(&query_embedding, 50).await?;

    if chunk_metas.is_empty() {
        println!("{}", "No results found.".yellow());
        return Ok(());
    }

    // Fetch content from SQLite
    let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
    let contents = content_store.get_chunks(&chunk_ids)?;
    let content_map: std::collections::HashMap<String, String> = contents.into_iter().collect();

    // Combine metadata + content into SearchResult
    let results: Vec<SearchResult> = chunk_metas
        .into_iter()
        .filter_map(|meta| {
            let content = content_map.get(&meta.id)?.clone();
            Some(SearchResult {
                id: meta.id,
                source_id: meta.source_id,
                title: meta.title,
                content,
                file_path: meta.file_path,
                line_start: meta.line_start,
                score: meta.score,
            })
        })
        .collect();

    // Filter and rerank
    let results = search_engine.filter_results(results);
    let results = search_engine.rerank(results, query, 5);

    if results.is_empty() {
        println!("{}", "No results found.".yellow());
        return Ok(());
    }

    let results: Vec<_> = results;

    for (i, result) in results.iter().enumerate() {
        println!(
            "  {}. {} {}",
            (i + 1).to_string().cyan().bold(),
            format!("[{:.2}]", result.score).dimmed(),
            result.title.as_deref().unwrap_or("Untitled").white().bold()
        );

        if let Some(ref file_path) = result.file_path {
            println!("     {}", file_path.dimmed());
        }

        // Show preview (first 150 chars)
        let preview: String = result
            .content
            .chars()
            .take(150)
            .collect::<String>()
            .replace('\n', " ");
        println!("     {}", preview.dimmed());
        println!();
    }

    Ok(())
}

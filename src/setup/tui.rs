//! TUI dashboard for setup wizard
//!
//! A polished, centered terminal UI for the first-run experience.

use super::download::{ModelDownloader, ModelTask};
use crate::config::Config;
use anyhow::Result;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{self, disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Padding, Paragraph},
    Frame, Terminal,
};
use std::{
    io::stdout,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// Fixed width for the dashboard (characters)
const DASHBOARD_WIDTH: u16 = 62;

/// Tips shown during download
const TIPS: &[&str] = &[
    "Everything runs locally. Your data never leaves your machine.",
    "Search works by meaning, not just keywords.",
    "Named after Avatar's Eywa - the network that stores all memory.",
    "Documents are split into chunks for precise retrieval.",
    "The reranker refines results using cross-encoder scoring.",
];

/// Setup wizard TUI
pub struct SetupWizard {
    config: Config,
}

/// Wizard state shared between TUI and download tasks
#[derive(Debug)]
struct WizardState {
    phase: Phase,
    embedding_task: Option<ModelTask>,
    reranker_task: Option<ModelTask>,
    current_speed: f64,
    start_time: Option<Instant>,
    error: Option<String>,
    tip_index: usize,
    last_tip_change: Instant,
}

impl WizardState {
    fn new() -> Self {
        Self {
            phase: Phase::Starting,
            embedding_task: None,
            reranker_task: None,
            current_speed: 0.0,
            start_time: None,
            error: None,
            tip_index: 0,
            last_tip_change: Instant::now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Phase {
    Starting,
    Downloading,
    Complete,
    Error,
}

impl SetupWizard {
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn run(&mut self) -> Result<()> {
        // Check if we have a real TTY
        use std::io::IsTerminal;
        if !std::io::stdout().is_terminal() {
            // Fallback to simple progress for non-interactive environments
            return self.run_simple();
        }

        // Use inline mode (change to run_fullscreen() for alternate screen mode)
        return self.run_inline();
    }

    /// Fullscreen TUI mode (alternate screen)
    #[allow(dead_code)]
    fn run_fullscreen(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
        terminal.clear()?;

        let state = Arc::new(Mutex::new(WizardState::new()));

        // Run the download in a separate thread with its own runtime
        let state_clone = Arc::clone(&state);
        let config = self.config.clone();

        let download_handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(run_downloads(state_clone, config))
        });

        // TUI loop
        let result = self.run_tui_loop(&mut terminal, &state);

        // Cleanup terminal
        disable_raw_mode()?;
        stdout().execute(LeaveAlternateScreen)?;

        // Wait for download thread
        if let Err(e) = download_handle.join() {
            eprintln!("Download thread panicked: {:?}", e);
        }

        result
    }

    /// Simple fallback for non-TTY environments
    fn run_simple(&mut self) -> Result<()> {
        println!("Downloading models...\n");

        // Use existing runtime if available, otherwise create new one
        let result = if let Ok(handle) = tokio::runtime::Handle::try_current() {
            std::thread::scope(|s| {
                s.spawn(|| {
                    handle.block_on(self.run_simple_async())
                }).join().unwrap()
            })
        } else {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(self.run_simple_async())
        };

        result
    }

    async fn run_simple_async(&self) -> Result<()> {
        let downloader = ModelDownloader::new();

        // Download embedding model
        println!("  {} ({}MB)", self.config.embedding_model.name(), self.config.embedding_model.size_mb());
        let embedding_task = downloader.create_tasks(&self.config.embedding_model).await?;
        let embedding_dir = downloader.model_cache_dir(&embedding_task.repo_id);
        let embedding_commit = embedding_task.commit_hash.clone();

        let mut task = embedding_task;
        for file in &mut task.files {
            if file.done {
                println!("    {} (cached)", file.name);
                continue;
            }
            print!("    {}...", file.name);
            std::io::Write::flush(&mut std::io::stdout())?;
            downloader
                .download_file(file, &embedding_dir, embedding_commit.as_deref(), |_| {})
                .await?;
            println!(" done");
        }

        // Download reranker model
        println!("  {} ({}MB)", self.config.reranker_model.name(), self.config.reranker_model.size_mb());
        let reranker_task = downloader.create_tasks(&self.config.reranker_model).await?;
        let reranker_dir = downloader.model_cache_dir(&reranker_task.repo_id);
        let reranker_commit = reranker_task.commit_hash.clone();

        let mut task = reranker_task;
        for file in &mut task.files {
            if file.done {
                println!("    {} (cached)", file.name);
                continue;
            }
            print!("    {}...", file.name);
            std::io::Write::flush(&mut std::io::stdout())?;
            downloader
                .download_file(file, &reranker_dir, reranker_commit.as_deref(), |_| {})
                .await?;
            println!(" done");
        }

        println!("\nDownload complete!");
        Ok(())
    }

    /// Inline mode - progress updates in place without taking over screen
    fn run_inline(&mut self) -> Result<()> {
        use std::io::Write;

        let state = Arc::new(Mutex::new(WizardState::new()));

        // Print initial structure
        println!();
        println!("  \x1b[1mDownloading Models\x1b[0m");
        println!();
        println!("  {}                                    0 B    0%", self.config.embedding_model.name());  // emb name
        println!("  \x1b[90m{}\x1b[0m", "━".repeat(54));  // emb bar
        println!();  // spacer
        println!("  {}                                    0 B    0%", self.config.reranker_model.name());  // rer name
        println!("  \x1b[90m{}\x1b[0m", "━".repeat(54));  // rer bar
        println!();
        println!("  \x1b[90mTotal: 0 B / 0 B    ETA: --\x1b[0m");
        println!();

        // Lines we print in the loop: emb name, emb bar, spacer, rer name, rer bar, blank, total, blank = 8
        const LINES_BACK: u16 = 8;

        // Run downloads in separate thread
        let state_clone = Arc::clone(&state);
        let config = self.config.clone();

        let download_handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(run_downloads(state_clone, config))
        });

        // Update loop
        loop {
            std::thread::sleep(Duration::from_millis(50));

            let s = state.lock().unwrap();

            // Move cursor up to update lines
            stdout().execute(cursor::MoveUp(LINES_BACK))?;

            // Embedding line
            const BAR_WIDTH: usize = 54;
            if let Some(ref task) = s.embedding_task {
                let downloaded: u64 = task.files.iter().map(|f| f.downloaded_bytes).sum();
                let total = task.size_mb as u64 * 1024 * 1024;
                let percent = if total > 0 { (downloaded * 100 / total) as u16 } else { 0 };
                let is_done = task.files.iter().all(|f| f.done);

                // Name and status line (right-aligned to bar width)
                stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
                let status = if is_done {
                    format!("\x1b[32m✓\x1b[0m")
                } else {
                    format!("\x1b[36m{:>3}%\x1b[0m", percent)
                };
                let size_str = format_bytes(downloaded);
                let right_part = format!("{}  {}", size_str, status);
                let left_pad = BAR_WIDTH.saturating_sub(task.name.len()).saturating_sub(right_part.len() - 9); // -9 for ANSI codes
                println!("  {}{}{}", task.name, " ".repeat(left_pad), right_part);

                // Progress bar
                stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
                let filled = (BAR_WIDTH * percent as usize) / 100;
                let empty = BAR_WIDTH - filled;
                let bar_color = if is_done { "\x1b[32m" } else { "\x1b[36m" };
                println!("  {}{}\x1b[0m\x1b[90m{}\x1b[0m", bar_color, "━".repeat(filled), "━".repeat(empty));
            } else {
                // Show model name with "Loading..." status while waiting for task
                stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
                let name = self.config.embedding_model.name();
                let right_part = "\x1b[90mLoading...\x1b[0m";
                let left_pad = BAR_WIDTH.saturating_sub(name.len()).saturating_sub(10);
                println!("  {}{}{}", name, " ".repeat(left_pad), right_part);
                stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
                println!("  \x1b[90m{}\x1b[0m", "━".repeat(BAR_WIDTH));
            }

            // Spacer between models
            stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
            println!();

            // Reranker line
            if let Some(ref task) = s.reranker_task {
                let downloaded: u64 = task.files.iter().map(|f| f.downloaded_bytes).sum();
                let total = task.size_mb as u64 * 1024 * 1024;
                let percent = if total > 0 { (downloaded * 100 / total) as u16 } else { 0 };
                let is_done = task.files.iter().all(|f| f.done);

                // Name and status line (right-aligned to bar width)
                stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
                let status = if is_done {
                    format!("\x1b[32m✓\x1b[0m")
                } else {
                    format!("\x1b[36m{:>3}%\x1b[0m", percent)
                };
                let size_str = format_bytes(downloaded);
                let right_part = format!("{}  {}", size_str, status);
                let left_pad = BAR_WIDTH.saturating_sub(task.name.len()).saturating_sub(right_part.len() - 9); // -9 for ANSI codes
                println!("  {}{}{}", task.name, " ".repeat(left_pad), right_part);

                // Progress bar
                stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
                let filled = (BAR_WIDTH * percent as usize) / 100;
                let empty = BAR_WIDTH - filled;
                let bar_color = if is_done { "\x1b[32m" } else { "\x1b[36m" };
                println!("  {}{}\x1b[0m\x1b[90m{}\x1b[0m", bar_color, "━".repeat(filled), "━".repeat(empty));
            } else {
                // Show model name with "Loading..." status while waiting for task
                stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
                let name = self.config.reranker_model.name();
                let right_part = "\x1b[90mLoading...\x1b[0m";
                let left_pad = BAR_WIDTH.saturating_sub(name.len()).saturating_sub(10);
                println!("  {}{}{}", name, " ".repeat(left_pad), right_part);
                stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
                println!("  \x1b[90m{}\x1b[0m", "━".repeat(BAR_WIDTH));
            }

            // Empty line
            stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
            println!();

            // Total line
            let (total_downloaded, total_size) = if let (Some(ref emb), Some(ref rer)) =
                (&s.embedding_task, &s.reranker_task)
            {
                let downloaded = emb.files.iter().map(|f| f.downloaded_bytes).sum::<u64>()
                    + rer.files.iter().map(|f| f.downloaded_bytes).sum::<u64>();
                let size = (emb.size_mb + rer.size_mb) as u64 * 1024 * 1024;
                (downloaded, size)
            } else {
                (0, 1)
            };

            let eta = if s.current_speed > 0.0 {
                let remaining = total_size.saturating_sub(total_downloaded) as f64;
                let secs = (remaining / s.current_speed) as u64;
                if secs < 60 { format!("{}s", secs) } else { format!("{}m {}s", secs / 60, secs % 60) }
            } else {
                "--".to_string()
            };

            stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
            println!("  \x1b[90mTotal: {} / {}    ETA: {}\x1b[0m", format_bytes(total_downloaded), format_bytes(total_size), eta);

            // Empty line (stay here for next iteration)
            stdout().execute(terminal::Clear(terminal::ClearType::CurrentLine))?;
            println!();

            std::io::stdout().flush()?;

            // Check if done
            if s.phase == Phase::Complete || s.phase == Phase::Error {
                break;
            }

            drop(s);
        }

        // Wait for download thread
        if let Err(e) = download_handle.join() {
            eprintln!("Download thread panicked: {:?}", e);
        }

        // Check for errors
        let s = state.lock().unwrap();
        if let Some(ref err) = s.error {
            anyhow::bail!("{}", err);
        }

        Ok(())
    }

    fn run_tui_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
        state: &Arc<Mutex<WizardState>>,
    ) -> Result<()> {
        loop {
            // Update tip periodically
            {
                let mut s = state.lock().unwrap();
                if s.last_tip_change.elapsed() > Duration::from_secs(5) {
                    s.tip_index = (s.tip_index + 1) % TIPS.len();
                    s.last_tip_change = Instant::now();
                }
            }

            // Draw
            let state_clone = Arc::clone(state);
            terminal.draw(|frame| draw(frame, &state_clone, &self.config))?;

            // Check if done
            {
                let s = state.lock().unwrap();
                if s.phase == Phase::Complete || s.phase == Phase::Error {
                    // Show final state for a moment
                    drop(s);
                    std::thread::sleep(Duration::from_millis(1000));
                    break;
                }
            }

            // Handle input (non-blocking)
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => {
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Check for errors
        let s = state.lock().unwrap();
        if let Some(ref err) = s.error {
            anyhow::bail!("{}", err);
        }

        Ok(())
    }
}

fn draw(frame: &mut Frame, state: &Arc<Mutex<WizardState>>, config: &Config) {
    let state = state.lock().unwrap();
    let area = frame.area();

    // Center the dashboard
    let dashboard_area = centered_rect(DASHBOARD_WIDTH, area.height.min(24), area);

    // Main layout
    let chunks = Layout::vertical([
        Constraint::Length(3), // Header
        Constraint::Length(5), // Config
        Constraint::Length(9), // Progress
        Constraint::Length(4), // Tips
        Constraint::Min(0),    // Spacer
    ])
    .split(dashboard_area);

    // Render each section
    render_header(frame, chunks[0]);
    render_config(frame, chunks[1], &state, config);
    render_progress(frame, chunks[2], &state);
    render_tips(frame, chunks[3], &state);
}

fn render_header(frame: &mut Frame, area: Rect) {
    let header = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("EYWA SETUP", Style::default().bold()),
            Span::raw(" ".repeat(38)),
            Span::styled(
                format!("v{}", env!("CARGO_PKG_VERSION")),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );

    frame.render_widget(header, area);
}

fn render_config(frame: &mut Frame, area: Rect, state: &WizardState, config: &Config) {
    let embedding_name = state
        .embedding_task
        .as_ref()
        .map(|t| t.name.clone())
        .unwrap_or_else(|| config.embedding_model.name().to_string());

    let reranker_name = state
        .reranker_task
        .as_ref()
        .map(|t| t.name.clone())
        .unwrap_or_else(|| config.reranker_model.name().to_string());

    let config_text = vec![
        Line::from(vec![Span::styled(
            "  Configuration",
            Style::default().fg(Color::DarkGray),
        )]),
        Line::from(vec![
            Span::raw("  ├─ Embedding:  "),
            Span::styled(&embedding_name, Style::default().fg(Color::White)),
            Span::styled(
                format!(" ({}d)", config.embedding_model.dimensions()),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::raw("  └─ Reranker:   "),
            Span::styled(&reranker_name, Style::default().fg(Color::White)),
        ]),
    ];

    let config_widget = Paragraph::new(config_text).block(
        Block::default()
            .borders(Borders::LEFT | Borders::RIGHT)
            .border_style(Style::default().fg(Color::DarkGray))
            .padding(Padding::vertical(1)),
    );

    frame.render_widget(config_widget, area);
}

fn render_progress(frame: &mut Frame, area: Rect, state: &WizardState) {
    let inner = Block::default()
        .borders(Borders::LEFT | Borders::RIGHT)
        .border_style(Style::default().fg(Color::DarkGray))
        .inner(area);

    frame.render_widget(
        Block::default()
            .borders(Borders::LEFT | Borders::RIGHT)
            .border_style(Style::default().fg(Color::DarkGray)),
        area,
    );

    // Progress section layout
    let chunks = Layout::vertical([
        Constraint::Length(1), // Title
        Constraint::Length(1), // Spacer
        Constraint::Length(2), // Embedding progress
        Constraint::Length(2), // Reranker progress
        Constraint::Length(1), // Spacer
        Constraint::Length(1), // Total
    ])
    .split(inner);

    // Title
    let title = match state.phase {
        Phase::Starting => "  Preparing downloads...",
        Phase::Downloading => "  Downloading Models",
        Phase::Complete => "  Setup Complete",
        Phase::Error => "  Error",
    };
    let title_style = match state.phase {
        Phase::Complete => Style::default().fg(Color::Green),
        Phase::Error => Style::default().fg(Color::Red),
        _ => Style::default().fg(Color::DarkGray),
    };
    frame.render_widget(Paragraph::new(title).style(title_style), chunks[0]);

    // Embedding progress
    if let Some(ref task) = state.embedding_task {
        render_model_progress(frame, chunks[2], task);
    } else {
        frame.render_widget(
            Paragraph::new("  Loading...").style(Style::default().fg(Color::DarkGray)),
            chunks[2],
        );
    }

    // Reranker progress
    if let Some(ref task) = state.reranker_task {
        render_model_progress(frame, chunks[3], task);
    } else {
        frame.render_widget(
            Paragraph::new("  Loading...").style(Style::default().fg(Color::DarkGray)),
            chunks[3],
        );
    }

    // Total progress
    let (total_downloaded, total_size) = if let (Some(ref emb), Some(ref rer)) =
        (&state.embedding_task, &state.reranker_task)
    {
        let downloaded = emb
            .files
            .iter()
            .map(|f| f.downloaded_bytes)
            .sum::<u64>()
            + rer.files.iter().map(|f| f.downloaded_bytes).sum::<u64>();
        let size = (emb.size_mb + rer.size_mb) as u64 * 1024 * 1024;
        (downloaded, size)
    } else {
        (0, 1) // Avoid division by zero
    };

    let eta = if state.current_speed > 0.0 {
        let remaining = total_size.saturating_sub(total_downloaded) as f64;
        let secs = (remaining / state.current_speed) as u64;
        if secs < 60 {
            format!("{}s", secs)
        } else {
            format!("{}m {}s", secs / 60, secs % 60)
        }
    } else {
        "--".to_string()
    };

    let total_line = format!(
        "  Total: {} / {}    ETA: {}",
        format_bytes(total_downloaded),
        format_bytes(total_size),
        eta
    );
    frame.render_widget(
        Paragraph::new(total_line).style(Style::default().fg(Color::DarkGray)),
        chunks[5],
    );
}

fn render_model_progress(frame: &mut Frame, area: Rect, task: &ModelTask) {
    let chunks = Layout::vertical([
        Constraint::Length(1), // Name + percentage
        Constraint::Length(1), // Progress bar
    ])
    .split(area);

    // Calculate progress
    let downloaded: u64 = task.files.iter().map(|f| f.downloaded_bytes).sum();
    let total: u64 = task.size_mb as u64 * 1024 * 1024;
    let percent = if total > 0 {
        ((downloaded as f64 / total as f64) * 100.0) as u16
    } else {
        0
    };

    let is_done = task.files.iter().all(|f| f.done);

    // Name line
    let status = if is_done {
        Span::styled(" ✓", Style::default().fg(Color::Green))
    } else {
        Span::styled(format!(" {:>3}%", percent), Style::default().fg(Color::Cyan))
    };

    let name_line = Line::from(vec![
        Span::raw("  "),
        Span::styled(&task.name, Style::default().fg(Color::White)),
        Span::raw(" ".repeat(28_usize.saturating_sub(task.name.len()))),
        Span::styled(format_bytes(downloaded), Style::default().fg(Color::DarkGray)),
        status,
    ]);
    frame.render_widget(Paragraph::new(name_line), chunks[0]);

    // Progress bar
    let bar_width = (area.width as usize).saturating_sub(4);
    let filled = (bar_width * percent as usize) / 100;
    let empty = bar_width.saturating_sub(filled);

    let bar_style = if is_done {
        Style::default().fg(Color::Green)
    } else {
        Style::default().fg(Color::Cyan)
    };

    let bar = Line::from(vec![
        Span::raw("  "),
        Span::styled("━".repeat(filled), bar_style),
        Span::styled("━".repeat(empty), Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(bar), chunks[1]);
}

fn render_tips(frame: &mut Frame, area: Rect, state: &WizardState) {
    let tip = TIPS.get(state.tip_index).unwrap_or(&TIPS[0]);

    let tips = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(*tip, Style::default().fg(Color::DarkGray).italic()),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray)),
    );

    frame.render_widget(tips, area);
}

/// Run the actual downloads
async fn run_downloads(state: Arc<Mutex<WizardState>>, config: Config) -> Result<()> {
    let downloader = ModelDownloader::new();

    // Create tasks (fetches commit hashes)
    let embedding_task = downloader.create_tasks(&config.embedding_model).await?;
    let reranker_task = downloader.create_tasks(&config.reranker_model).await?;

    // Update state with tasks
    {
        let mut s = state.lock().unwrap();
        s.embedding_task = Some(embedding_task.clone());
        s.reranker_task = Some(reranker_task.clone());
        s.phase = Phase::Downloading;
        s.start_time = Some(Instant::now());
    }

    // Download embedding model
    let embedding_dir = downloader.model_cache_dir(&embedding_task.repo_id);
    let embedding_commit = embedding_task.commit_hash.clone();
    {
        let mut task = embedding_task;

        for file in &mut task.files {
            if file.done {
                continue;
            }

            let state_clone = Arc::clone(&state);
            let file_name = file.name.clone();

            let result = downloader
                .download_file(
                    file,
                    &embedding_dir,
                    embedding_commit.as_deref(),
                    |progress| {
                        let mut s = state_clone.lock().unwrap();
                        // Update the specific file in embedding_task
                        if let Some(ref mut task) = s.embedding_task {
                            if let Some(f) = task.files.iter_mut().find(|f| f.name == file_name) {
                                f.downloaded_bytes = progress.bytes_downloaded;
                                f.size_bytes = progress.total_bytes;
                                f.done = progress.done;
                            }
                        }

                        // Calculate speed
                        if let Some(start) = s.start_time {
                            let elapsed = start.elapsed().as_secs_f64();
                            let emb_downloaded: u64 = s
                                .embedding_task
                                .as_ref()
                                .map(|t| t.files.iter().map(|f| f.downloaded_bytes).sum())
                                .unwrap_or(0);
                            let rer_downloaded: u64 = s
                                .reranker_task
                                .as_ref()
                                .map(|t| t.files.iter().map(|f| f.downloaded_bytes).sum())
                                .unwrap_or(0);
                            let total_downloaded = emb_downloaded + rer_downloaded;
                            if elapsed > 0.0 {
                                s.current_speed = total_downloaded as f64 / elapsed;
                            }
                        }
                    },
                )
                .await;

            if let Err(e) = result {
                let mut s = state.lock().unwrap();
                s.phase = Phase::Error;
                s.error = Some(format!("Failed to download {}: {}", file.name, e));
                return Err(e);
            }
        }

        // Mark embedding as complete in state
        let mut s = state.lock().unwrap();
        if let Some(ref mut t) = s.embedding_task {
            for f in &mut t.files {
                f.done = true;
            }
        }
    }

    // Download reranker model
    let reranker_dir = downloader.model_cache_dir(&reranker_task.repo_id);
    let reranker_commit = reranker_task.commit_hash.clone();
    {
        let mut task = reranker_task;

        for file in &mut task.files {
            if file.done {
                continue;
            }

            let state_clone = Arc::clone(&state);
            let file_name = file.name.clone();

            let result = downloader
                .download_file(
                    file,
                    &reranker_dir,
                    reranker_commit.as_deref(),
                    |progress| {
                        let mut s = state_clone.lock().unwrap();
                        // Update the specific file in reranker_task
                        if let Some(ref mut task) = s.reranker_task {
                            if let Some(f) = task.files.iter_mut().find(|f| f.name == file_name) {
                                f.downloaded_bytes = progress.bytes_downloaded;
                                f.size_bytes = progress.total_bytes;
                                f.done = progress.done;
                            }
                        }

                        // Calculate speed
                        if let Some(start) = s.start_time {
                            let elapsed = start.elapsed().as_secs_f64();
                            let emb_downloaded: u64 = s
                                .embedding_task
                                .as_ref()
                                .map(|t| t.files.iter().map(|f| f.downloaded_bytes).sum())
                                .unwrap_or(0);
                            let rer_downloaded: u64 = s
                                .reranker_task
                                .as_ref()
                                .map(|t| t.files.iter().map(|f| f.downloaded_bytes).sum())
                                .unwrap_or(0);
                            let total_downloaded = emb_downloaded + rer_downloaded;
                            if elapsed > 0.0 {
                                s.current_speed = total_downloaded as f64 / elapsed;
                            }
                        }
                    },
                )
                .await;

            if let Err(e) = result {
                let mut s = state.lock().unwrap();
                s.phase = Phase::Error;
                s.error = Some(format!("Failed to download {}: {}", file.name, e));
                return Err(e);
            }
        }

        // Mark reranker as complete in state
        let mut s = state.lock().unwrap();
        if let Some(ref mut t) = s.reranker_task {
            for f in &mut t.files {
                f.done = true;
            }
        }
    }

    // Mark complete
    {
        let mut s = state.lock().unwrap();
        s.phase = Phase::Complete;
    }

    Ok(())
}

/// Center a rect within another rect
fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    Rect::new(x, y, width.min(area.width), height.min(area.height))
}

/// Format bytes as human-readable
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

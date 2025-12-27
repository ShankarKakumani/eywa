//! Persistent job queue for background document processing
//!
//! Stores jobs and pending documents in SQLite so they survive server restarts.
//! Users can upload documents, close browser, and come back later.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::types::{DocStatus, DocumentInput, JobProgress, JobStatus, PendingDoc};

/// Document info for status API (without content)
#[derive(Debug, Clone, serde::Serialize)]
pub struct PendingDocInfo {
    pub id: String,
    pub title: Option<String>,
    pub file_path: Option<String>,
    pub status: DocStatus,
    pub error: Option<String>,
}

/// Persistent job queue backed by SQLite
pub struct JobQueue {
    conn: Connection,
}

impl JobQueue {
    /// Open or create a job queue at the given database path
    pub fn open(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open job queue at {:?}", db_path))?;

        let queue = Self { conn };
        queue.init_schema()?;
        queue.recover_processing()?;

        Ok(queue)
    }

    /// Initialize database schema
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                total_docs INTEGER NOT NULL,
                completed_docs INTEGER DEFAULT 0,
                failed_docs INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                current_doc TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS pending_docs (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                file_path TEXT,
                status TEXT DEFAULT 'pending',
                error TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_pending_docs_job
                ON pending_docs(job_id);

            CREATE INDEX IF NOT EXISTS idx_pending_docs_status
                ON pending_docs(status);

            PRAGMA foreign_keys = ON;
            ",
        )?;

        Ok(())
    }

    /// Reset any docs that were "processing" back to "pending" (server restart recovery)
    fn recover_processing(&self) -> Result<()> {
        self.conn.execute(
            "UPDATE pending_docs SET status = 'pending' WHERE status = 'processing'",
            [],
        )?;
        self.conn.execute(
            "UPDATE jobs SET status = 'pending' WHERE status = 'processing'",
            [],
        )?;
        Ok(())
    }

    /// Queue documents for processing, returns job_id
    pub fn queue_documents(&mut self, source_id: &str, documents: Vec<DocumentInput>) -> Result<String> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        let total_docs = documents.len() as u32;

        // Insert job
        self.conn.execute(
            "INSERT INTO jobs (id, source_id, total_docs, status, created_at)
             VALUES (?1, ?2, ?3, 'pending', ?4)",
            params![job_id, source_id, total_docs, now],
        )?;

        // Insert pending docs
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO pending_docs (id, job_id, source_id, title, content, file_path, status, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7)",
            )?;

            for doc in documents {
                let doc_id = uuid::Uuid::new_v4().to_string();
                stmt.execute(params![
                    doc_id,
                    job_id,
                    source_id,
                    doc.title,
                    doc.content,
                    doc.file_path,
                    now
                ])?;
            }
        }
        tx.commit()?;

        Ok(job_id)
    }

    /// Get the next pending document to process
    pub fn get_next_pending(&mut self) -> Result<Option<PendingDoc>> {
        let doc: Option<(String, String, String, Option<String>, String, Option<String>, String)> = self
            .conn
            .query_row(
                "SELECT id, job_id, source_id, title, content, file_path, created_at
                 FROM pending_docs WHERE status = 'pending' LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                },
            )
            .optional()?;

        let Some((id, job_id, source_id, title, content, file_path, created_at)) = doc else {
            return Ok(None);
        };

        // Mark as processing
        self.conn.execute(
            "UPDATE pending_docs SET status = 'processing' WHERE id = ?1",
            params![id],
        )?;
        self.conn.execute(
            "UPDATE jobs SET status = 'processing', current_doc = ?2 WHERE id = ?1",
            params![job_id, title],
        )?;

        Ok(Some(PendingDoc {
            id,
            job_id,
            source_id,
            title,
            content,
            file_path,
            status: DocStatus::Processing,
            error: None,
            created_at,
        }))
    }

    /// Get a batch of pending documents (up to limit) for batch processing
    pub fn get_pending_batch(&mut self, limit: usize) -> Result<Vec<PendingDoc>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, job_id, source_id, title, content, file_path, created_at
             FROM pending_docs WHERE status = 'pending' LIMIT ?1",
        )?;

        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, String>(6)?,
            ))
        })?;

        let mut batch = Vec::new();
        let mut doc_ids = Vec::new();
        let mut job_ids = std::collections::HashSet::new();

        for row in rows {
            let (id, job_id, source_id, title, content, file_path, created_at) = row?;
            doc_ids.push(id.clone());
            job_ids.insert(job_id.clone());
            batch.push(PendingDoc {
                id,
                job_id,
                source_id,
                title,
                content,
                file_path,
                status: DocStatus::Processing,
                error: None,
                created_at,
            });
        }

        // Mark all as processing
        if !doc_ids.is_empty() {
            let placeholders: Vec<&str> = doc_ids.iter().map(|_| "?").collect();
            let query = format!(
                "UPDATE pending_docs SET status = 'processing' WHERE id IN ({})",
                placeholders.join(",")
            );
            self.conn.execute(&query, rusqlite::params_from_iter(doc_ids.iter()))?;

            // Update job statuses
            for job_id in job_ids {
                self.conn.execute(
                    "UPDATE jobs SET status = 'processing' WHERE id = ?1",
                    params![job_id],
                )?;
            }
        }

        Ok(batch)
    }

    /// Mark a document as currently processing (for granular status updates)
    pub fn mark_processing(&mut self, doc_id: &str) -> Result<()> {
        // Get job_id and title
        let row: Option<(String, Option<String>)> = self
            .conn
            .query_row(
                "SELECT job_id, title FROM pending_docs WHERE id = ?1",
                params![doc_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        let Some((job_id, title)) = row else {
            return Ok(());
        };

        // Update doc status to processing
        self.conn.execute(
            "UPDATE pending_docs SET status = 'processing' WHERE id = ?1",
            params![doc_id],
        )?;

        // Update job's current_doc for display
        self.conn.execute(
            "UPDATE jobs SET status = 'processing', current_doc = ?2 WHERE id = ?1",
            params![job_id, title],
        )?;

        Ok(())
    }

    /// Mark a document as completed
    pub fn mark_completed(&mut self, doc_id: &str) -> Result<()> {
        // Get job_id first
        let job_id: Option<String> = self
            .conn
            .query_row(
                "SELECT job_id FROM pending_docs WHERE id = ?1",
                params![doc_id],
                |row| row.get(0),
            )
            .optional()?;

        let Some(job_id) = job_id else {
            return Ok(());
        };

        // Update doc status
        self.conn.execute(
            "UPDATE pending_docs SET status = 'done' WHERE id = ?1",
            params![doc_id],
        )?;

        // Update job counts
        self.conn.execute(
            "UPDATE jobs SET completed_docs = completed_docs + 1, current_doc = NULL WHERE id = ?1",
            params![job_id],
        )?;

        self.check_job_completion(&job_id)?;
        Ok(())
    }

    /// Mark a document as failed
    pub fn mark_failed(&mut self, doc_id: &str, error: &str) -> Result<()> {
        // Get job_id first
        let job_id: Option<String> = self
            .conn
            .query_row(
                "SELECT job_id FROM pending_docs WHERE id = ?1",
                params![doc_id],
                |row| row.get(0),
            )
            .optional()?;

        let Some(job_id) = job_id else {
            return Ok(());
        };

        // Update doc status
        self.conn.execute(
            "UPDATE pending_docs SET status = 'failed', error = ?2 WHERE id = ?1",
            params![doc_id, error],
        )?;

        // Update job counts
        self.conn.execute(
            "UPDATE jobs SET failed_docs = failed_docs + 1, current_doc = NULL WHERE id = ?1",
            params![job_id],
        )?;

        self.check_job_completion(&job_id)?;
        Ok(())
    }

    /// Check if a job is complete and update its status
    fn check_job_completion(&self, job_id: &str) -> Result<()> {
        let job: Option<(u32, u32, u32)> = self
            .conn
            .query_row(
                "SELECT total_docs, completed_docs, failed_docs FROM jobs WHERE id = ?1",
                params![job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;

        let Some((total, completed, failed)) = job else {
            return Ok(());
        };

        let processed = completed + failed;
        if processed >= total {
            let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
            let status = if failed > 0 && completed == 0 {
                "failed"
            } else {
                "done"
            };
            self.conn.execute(
                "UPDATE jobs SET status = ?2, completed_at = ?3 WHERE id = ?1",
                params![job_id, status, now],
            )?;
        }

        Ok(())
    }

    /// Get job progress
    pub fn get_job(&self, job_id: &str) -> Result<Option<JobProgress>> {
        let job: Option<(String, String, String, u32, u32, u32, Option<String>, String, Option<String>)> = self
            .conn
            .query_row(
                "SELECT id, source_id, status, total_docs, completed_docs, failed_docs, current_doc, created_at, completed_at
                 FROM jobs WHERE id = ?1",
                params![job_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                        row.get(8)?,
                    ))
                },
            )
            .optional()?;

        let Some((id, source_id, status_str, total, completed, failed, current_doc, created_at, completed_at)) = job else {
            return Ok(None);
        };

        let status = status_str.parse().unwrap_or(JobStatus::Pending);

        Ok(Some(JobProgress {
            job_id: id,
            source_id,
            status,
            total,
            completed,
            failed,
            current_doc,
            created_at,
            completed_at,
        }))
    }

    /// Get all documents for a job (for per-doc status API)
    pub fn get_job_docs(&self, job_id: &str) -> Result<Vec<PendingDocInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, file_path, status, error
             FROM pending_docs WHERE job_id = ?1
             ORDER BY created_at ASC",
        )?;

        let rows = stmt.query_map(params![job_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })?;

        let mut docs = Vec::new();
        for row in rows {
            let (id, title, file_path, status_str, error) = row?;
            let status = status_str.parse().unwrap_or(DocStatus::Pending);
            docs.push(PendingDocInfo {
                id,
                title,
                file_path,
                status,
                error,
            });
        }

        Ok(docs)
    }

    /// List all jobs
    pub fn list_jobs(&self) -> Result<Vec<JobProgress>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_id, status, total_docs, completed_docs, failed_docs, current_doc, created_at, completed_at
             FROM jobs ORDER BY created_at DESC",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, u32>(3)?,
                row.get::<_, u32>(4)?,
                row.get::<_, u32>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, Option<String>>(8)?,
            ))
        })?;

        let mut jobs = Vec::new();
        for row in rows {
            let (id, source_id, status_str, total, completed, failed, current_doc, created_at, completed_at) = row?;
            let status = status_str.parse().unwrap_or(JobStatus::Pending);
            jobs.push(JobProgress {
                job_id: id,
                source_id,
                status,
                total,
                completed,
                failed,
                current_doc,
                created_at,
                completed_at,
            });
        }

        Ok(jobs)
    }

    /// Check if there are any pending documents
    pub fn has_pending(&self) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM pending_docs WHERE status = 'pending'",
            [],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Get pending count
    pub fn pending_count(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM pending_docs WHERE status = 'pending'",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Clean up completed jobs older than the specified duration
    pub fn cleanup_old_jobs(&mut self, max_age_secs: i64) -> Result<()> {
        let cutoff = chrono::Utc::now() - chrono::Duration::seconds(max_age_secs);
        let cutoff_str = cutoff.format("%Y-%m-%dT%H:%M:%SZ").to_string();

        // Delete old completed jobs (pending_docs deleted via CASCADE)
        self.conn.execute(
            "DELETE FROM jobs WHERE status IN ('done', 'failed') AND created_at < ?1",
            params![cutoff_str],
        )?;

        Ok(())
    }
}

/// Thread-safe job queue wrapper (uses std::sync::Mutex since SQLite isn't Sync)
pub type SharedJobQueue = Arc<Mutex<JobQueue>>;

pub fn create_job_queue(db_path: &Path) -> Result<SharedJobQueue> {
    let queue = JobQueue::open(db_path)?;
    Ok(Arc::new(Mutex::new(queue)))
}

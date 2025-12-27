use serde::{Deserialize, Serialize};

/// A document (stores full original content)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub file_path: Option<String>,
    pub created_at: String,
    pub chunk_count: u32,
}

/// Document metadata (without content, for listing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMeta {
    pub id: String,
    pub source_id: String,
    pub title: String,
    pub file_path: Option<String>,
    pub created_at: String,
    pub chunk_count: u32,
    pub content_length: usize,
}

/// A document chunk with its embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub source_id: String,
    pub title: Option<String>,
    pub content: String,
    pub file_path: Option<String>,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub content_hash: String,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub source_id: String,
    pub title: Option<String>,
    pub content: String,
    pub file_path: Option<String>,
    pub line_start: Option<u32>,
    pub score: f32,
}

/// Chunk metadata from vector search (content fetched separately from SQLite)
#[derive(Debug, Clone)]
pub struct ChunkMeta {
    pub id: String,
    pub document_id: String,
    pub source_id: String,
    pub title: Option<String>,
    pub file_path: Option<String>,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub score: f32,
}

/// Document metadata (for when content is fetched separately)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub id: String,
    pub source_id: String,
    pub title: String,
    pub file_path: Option<String>,
    pub created_at: String,
    pub chunk_count: u32,
    pub content_length: u32,
}

/// Document source (collection)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub doc_count: u64,
    pub chunk_count: u64,
    pub last_indexed: Option<String>,
}

/// Result of ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    pub source_id: String,
    pub files_processed: u32,
    pub chunks_created: u32,
    pub chunks_skipped: u32,
}

/// API search request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub source_id: Option<String>,
}

fn default_limit() -> usize {
    5
}

/// API search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub query: String,
    pub results: Vec<SearchResult>,
    pub count: usize,
}

/// Input document for ingestion (from API/paste)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInput {
    pub content: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub file_path: Option<String>,
}

/// API ingest request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestRequest {
    pub source_id: String,
    pub documents: Vec<DocumentInput>,
}

/// API ingest response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResponse {
    pub source_id: String,
    pub documents_created: u32,
    pub chunks_created: u32,
    pub chunks_skipped: u32,
    pub document_ids: Vec<String>,
}

/// API fetch URL request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchUrlRequest {
    pub url: String,
    #[serde(default)]
    pub source_id: Option<String>,
}

// ============================================================================
// Job Queue Types
// ============================================================================

/// Status of a pending document in the queue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocStatus {
    Pending,
    Processing,
    Done,
    Failed,
}

impl std::fmt::Display for DocStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocStatus::Pending => write!(f, "pending"),
            DocStatus::Processing => write!(f, "processing"),
            DocStatus::Done => write!(f, "done"),
            DocStatus::Failed => write!(f, "failed"),
        }
    }
}

impl std::str::FromStr for DocStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(DocStatus::Pending),
            "processing" => Ok(DocStatus::Processing),
            "done" => Ok(DocStatus::Done),
            "failed" => Ok(DocStatus::Failed),
            _ => Err(format!("Unknown status: {}", s)),
        }
    }
}

/// A pending document waiting to be processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingDoc {
    pub id: String,
    pub job_id: String,
    pub source_id: String,
    pub title: Option<String>,
    pub content: String,
    pub file_path: Option<String>,
    pub status: DocStatus,
    pub error: Option<String>,
    pub created_at: String,
}

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Processing,
    Done,
    Failed,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatus::Pending => write!(f, "pending"),
            JobStatus::Processing => write!(f, "processing"),
            JobStatus::Done => write!(f, "done"),
            JobStatus::Failed => write!(f, "failed"),
        }
    }
}

impl std::str::FromStr for JobStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(JobStatus::Pending),
            "processing" => Ok(JobStatus::Processing),
            "done" => Ok(JobStatus::Done),
            "failed" => Ok(JobStatus::Failed),
            _ => Err(format!("Unknown status: {}", s)),
        }
    }
}

/// An ingestion job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub source_id: String,
    pub total_docs: u32,
    pub completed_docs: u32,
    pub failed_docs: u32,
    pub status: JobStatus,
    pub current_doc: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

/// Response when queuing documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueResponse {
    pub job_id: String,
    pub docs_queued: u32,
    pub message: String,
}

/// Job progress response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobProgress {
    pub job_id: String,
    pub source_id: String,
    pub status: JobStatus,
    pub total: u32,
    pub completed: u32,
    pub failed: u32,
    pub current_doc: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_status_roundtrip() {
        for status in [DocStatus::Pending, DocStatus::Processing, DocStatus::Done, DocStatus::Failed] {
            let s = status.to_string();
            let parsed: DocStatus = s.parse().unwrap();
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_job_status_roundtrip() {
        for status in [JobStatus::Pending, JobStatus::Processing, JobStatus::Done, JobStatus::Failed] {
            let s = status.to_string();
            let parsed: JobStatus = s.parse().unwrap();
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_invalid_status_parse() {
        assert!("invalid".parse::<DocStatus>().is_err());
        assert!("invalid".parse::<JobStatus>().is_err());
    }

    #[test]
    fn test_default_limit() {
        let req: SearchRequest = serde_json::from_str(r#"{"query": "test"}"#).unwrap();
        assert_eq!(req.limit, 5);
    }
}

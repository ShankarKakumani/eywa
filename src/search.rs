//! Search engine with optional reranking
//!
//! Provides semantic search with configurable result filtering and neural reranking.

use crate::rerank::Reranker;
use crate::types::SearchResult;

/// Search engine configuration
pub struct SearchEngine {
    /// Minimum similarity score threshold (0.0 - 1.0)
    pub min_score: f32,
    /// Optional neural reranker for better accuracy
    pub reranker: Option<Reranker>,
}

impl SearchEngine {
    /// Create a new search engine with default settings (no reranker)
    pub fn new() -> Self {
        Self {
            min_score: 0.3,
            reranker: None,
        }
    }

    /// Create a new search engine with neural reranker
    pub fn with_reranker() -> anyhow::Result<Self> {
        Ok(Self {
            min_score: 0.3,
            reranker: Some(Reranker::new()?),
        })
    }

    /// Create a new search engine with custom minimum score
    pub fn with_min_score(min_score: f32) -> Self {
        Self {
            min_score,
            reranker: None,
        }
    }

    /// Filter results by minimum score
    pub fn filter_results(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        results
            .into_iter()
            .filter(|r| r.score >= self.min_score)
            .collect()
    }

    /// Rerank results using neural reranker if available, otherwise use keyword boost
    pub fn rerank(&self, mut results: Vec<SearchResult>, query: &str, limit: usize) -> Vec<SearchResult> {
        if let Some(ref reranker) = self.reranker {
            // Use neural reranker
            let documents: Vec<String> = results.iter().map(|r| r.content.clone()).collect();

            if let Ok(scores) = reranker.rerank(query, &documents) {
                // Update scores with reranker scores
                for (result, score) in results.iter_mut().zip(scores.iter()) {
                    result.score = *score;
                }

                // Re-sort by reranker score
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else {
            // Fall back to keyword reranking
            results = self.rerank_with_keywords(results, query);
        }

        results.into_iter().take(limit).collect()
    }

    /// Rerank results using a simple BM25-like scoring boost
    /// This gives a small boost to exact keyword matches
    pub fn rerank_with_keywords(&self, mut results: Vec<SearchResult>, query: &str) -> Vec<SearchResult> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        for result in &mut results {
            let content_lower = result.content.to_lowercase();
            let mut keyword_boost = 0.0f32;

            for term in &query_terms {
                if content_lower.contains(term) {
                    keyword_boost += 0.05; // Small boost per matching term
                }
            }

            // Cap the boost at 0.2
            result.score += keyword_boost.min(0.2);
        }

        // Re-sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Check if reranker is available
    pub fn has_reranker(&self) -> bool {
        self.reranker.is_some()
    }
}

impl Default for SearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            source_id: "test".to_string(),
            title: None,
            content: content.to_string(),
            file_path: None,
            line_start: None,
            score,
        }
    }

    #[test]
    fn test_filter_empty_results() {
        let engine = SearchEngine::new();
        let filtered = engine.filter_results(vec![]);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_all_below_threshold() {
        let engine = SearchEngine::new(); // min_score = 0.3
        let results = vec![
            make_result("1", "low", 0.1),
            make_result("2", "lower", 0.2),
        ];
        let filtered = engine.filter_results(results);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_keeps_above_threshold() {
        let engine = SearchEngine::new();
        let results = vec![
            make_result("1", "high", 0.8),
            make_result("2", "low", 0.1),
            make_result("3", "medium", 0.5),
        ];
        let filtered = engine.filter_results(results);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|r| r.score >= 0.3));
    }

    #[test]
    fn test_rerank_empty() {
        let engine = SearchEngine::new();
        let reranked = engine.rerank_with_keywords(vec![], "query");
        assert!(reranked.is_empty());
    }

    #[test]
    fn test_rerank_case_insensitive() {
        let engine = SearchEngine::new();
        // Start with equal scores so keyword boost determines order
        let results = vec![
            make_result("1", "RUST is great", 0.5),
            make_result("2", "python is nice", 0.5),
        ];
        let reranked = engine.rerank_with_keywords(results, "rust");
        assert_eq!(reranked[0].id, "1"); // RUST matched rust (case insensitive)
    }

    #[test]
    fn test_rerank_multiple_terms() {
        let engine = SearchEngine::new();
        let results = vec![
            make_result("1", "rust", 0.5),
            make_result("2", "rust programming language", 0.5),
        ];
        let reranked = engine.rerank_with_keywords(results, "rust programming");
        // Result 2 should be boosted more (has both terms)
        assert_eq!(reranked[0].id, "2");
    }

    #[test]
    fn test_custom_min_score() {
        let engine = SearchEngine::with_min_score(0.7);
        let results = vec![
            make_result("1", "high", 0.8),
            make_result("2", "medium", 0.5),
        ];
        let filtered = engine.filter_results(results);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "1");
    }

    #[test]
    fn test_has_reranker() {
        let engine = SearchEngine::new();
        assert!(!engine.has_reranker());
    }
}

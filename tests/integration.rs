//! Integration tests for Eywa

use eywa::{BM25Index, ContentStore, DevicePreference, Embedder, EmbeddingModel, IngestPipeline, Ingester, SearchEngine, VectorDB};
use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn test_embedder_creates_correct_dimensions() {
    let embedder = Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder");
    // Dimension depends on configured model (384 for MiniLM, 768 for BGE)
    assert!(embedder.dimension() > 0, "Should have positive dimensions");
}

#[test]
fn test_embedder_single_text() {
    let embedder = Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder");
    let embedding = embedder.embed("hello world").expect("Failed to embed");

    assert_eq!(embedding.len(), embedder.dimension());

    // Check that it's normalized (L2 norm should be ~1.0)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized, got norm={}", norm);
}

#[test]
fn test_embedder_batch() {
    let embedder = Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder");
    let texts = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
    let embeddings = embedder.embed_batch(&texts).expect("Failed to batch embed");

    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.len(), embedder.dimension());
    }
}

#[test]
fn test_embedder_similar_texts_have_high_similarity() {
    let embedder = Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder");

    let emb1 = embedder.embed("The cat sat on the mat").unwrap();
    let emb2 = embedder.embed("A cat is sitting on a mat").unwrap();
    let emb3 = embedder.embed("Quantum physics explains the universe").unwrap();

    // Cosine similarity (vectors are normalized, so dot product = cosine)
    let sim_similar: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    let sim_different: f32 = emb1.iter().zip(emb3.iter()).map(|(a, b)| a * b).sum();

    assert!(sim_similar > sim_different,
        "Similar texts should have higher similarity: {} vs {}", sim_similar, sim_different);
    assert!(sim_similar > 0.7, "Similar texts should have high similarity: {}", sim_similar);
}

#[test]
fn test_search_engine_filters_by_score() {
    let engine = SearchEngine::new();

    let results = vec![
        eywa::SearchResult {
            id: "1".to_string(),
            source_id: "test".to_string(),
            title: Some("High".to_string()),
            content: "content".to_string(),
            file_path: None,
            line_start: None,
            score: 0.8,
        },
        eywa::SearchResult {
            id: "2".to_string(),
            source_id: "test".to_string(),
            title: Some("Low".to_string()),
            content: "content".to_string(),
            file_path: None,
            line_start: None,
            score: 0.2, // Below threshold of 0.3
        },
    ];

    let filtered = engine.filter_results(results);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].id, "1");
}

#[test]
fn test_search_engine_keyword_reranking() {
    let engine = SearchEngine::new();

    let results = vec![
        eywa::SearchResult {
            id: "1".to_string(),
            source_id: "test".to_string(),
            title: None,
            content: "This is about dogs and cats".to_string(),
            file_path: None,
            line_start: None,
            score: 0.7,
        },
        eywa::SearchResult {
            id: "2".to_string(),
            source_id: "test".to_string(),
            title: None,
            content: "Rust programming language is great".to_string(),
            file_path: None,
            line_start: None,
            score: 0.75,
        },
    ];

    // Search for "rust" should boost the second result
    let reranked = engine.rerank_with_keywords(results, "rust programming");

    assert_eq!(reranked[0].id, "2", "Result with keyword match should be first");
}

#[tokio::test]
async fn test_vectordb_create_and_search() {
    let dir = tempdir().expect("Failed to create temp dir");
    let data_path = dir.path();

    let embedder = Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder");
    let mut db = VectorDB::new(data_path.to_str().unwrap()).await.expect("Failed to create db");

    // Ingest a document
    let ingester = Ingester::new(&embedder);
    let docs = vec![eywa::DocumentInput {
        content: "Rust is a systems programming language focused on safety and performance.".to_string(),
        title: Some("Rust Overview".to_string()),
        file_path: None,
        is_pdf: false,
    }];

    let result = ingester.ingest_documents(&mut db, data_path, "test-source", docs).await
        .expect("Failed to ingest");

    assert_eq!(result.documents_created, 1);
    assert!(result.chunks_created >= 1);

    // Search for it
    let query_embedding = embedder.embed("What is Rust?").expect("Failed to embed query");
    let chunk_metas = db.search(&query_embedding, 5).await.expect("Failed to search");

    assert!(!chunk_metas.is_empty(), "Should find at least one result");

    // Fetch content from SQLite to verify
    let content_store = ContentStore::open(&data_path.join("content.db")).expect("Failed to open content store");
    let chunk_ids: Vec<&str> = chunk_metas.iter().map(|c| c.id.as_str()).collect();
    let contents = content_store.get_chunks(&chunk_ids).expect("Failed to get chunks");
    let content_map: std::collections::HashMap<String, String> = contents.into_iter().collect();

    let first_content = content_map.get(&chunk_metas[0].id).expect("Content should exist");
    assert!(first_content.contains("Rust"), "Result should contain 'Rust'");
}

#[tokio::test]
async fn test_deduplication() {
    let dir = tempdir().expect("Failed to create temp dir");
    let data_path = dir.path();

    let embedder = Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder");
    let mut db = VectorDB::new(data_path.to_str().unwrap()).await.expect("Failed to create db");
    let ingester = Ingester::new(&embedder);

    let docs = vec![eywa::DocumentInput {
        content: "This is a test document.".to_string(),
        title: Some("Test".to_string()),
        file_path: None,
        is_pdf: false,
    }];

    // Ingest same content twice
    let result1 = ingester.ingest_documents(&mut db, data_path, "source1", docs.clone()).await.unwrap();
    let result2 = ingester.ingest_documents(&mut db, data_path, "source2", docs).await.unwrap();

    assert_eq!(result1.chunks_created, 1);
    assert_eq!(result2.chunks_skipped, 1, "Duplicate chunk should be skipped");
}

#[tokio::test]
async fn test_source_management() {
    let dir = tempdir().expect("Failed to create temp dir");
    let data_path = dir.path();

    let embedder = Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder");
    let mut db = VectorDB::new(data_path.to_str().unwrap()).await.expect("Failed to create db");
    let ingester = Ingester::new(&embedder);

    // Add documents to two sources
    let docs = vec![eywa::DocumentInput {
        content: "Document one content here.".to_string(),
        title: Some("Doc1".to_string()),
        file_path: None,
        is_pdf: false,
    }];

    ingester.ingest_documents(&mut db, data_path, "source-a", docs.clone()).await.unwrap();

    let docs2 = vec![eywa::DocumentInput {
        content: "Different document content.".to_string(),
        title: Some("Doc2".to_string()),
        file_path: None,
        is_pdf: false,
    }];
    ingester.ingest_documents(&mut db, data_path, "source-b", docs2).await.unwrap();

    // List sources
    let sources = db.list_sources().await.expect("Failed to list sources");
    assert_eq!(sources.len(), 2);

    // Delete one source
    db.delete_source("source-a").await.expect("Failed to delete source");

    let sources = db.list_sources().await.expect("Failed to list sources");
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].id, "source-b");
}

// ============================================================
// Hybrid Search Integration Tests
// ============================================================

#[tokio::test]
async fn test_ingest_pipeline_indexes_to_bm25() {
    let dir = tempdir().expect("Failed to create temp dir");
    let data_path = dir.path();

    let embedder = Arc::new(Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder"));
    let bm25_index = Arc::new(BM25Index::open(data_path).expect("Failed to create BM25 index"));
    let mut db = VectorDB::new(data_path.to_str().unwrap()).await.expect("Failed to create db");

    let pipeline = IngestPipeline::new(Arc::clone(&embedder), Arc::clone(&bm25_index));

    // Ingest documents with specific keywords (must be long enough to create chunks)
    let docs = vec![
        eywa::DocumentInput {
            content: "JWT authentication uses tokens for secure API access. JSON Web Tokens (JWT) are an open standard for securely transmitting information between parties as a JSON object. This information can be verified and trusted because it is digitally signed. JWTs can be signed using a secret or a public/private key pair.".to_string(),
            title: Some("Auth Guide".to_string()),
            file_path: None,
            is_pdf: false,
        },
        eywa::DocumentInput {
            content: "OAuth2 provides authorization framework for third-party apps. OAuth 2.0 is the industry-standard protocol for authorization. It focuses on client developer simplicity while providing specific authorization flows for web applications, desktop applications, mobile phones, and IoT devices.".to_string(),
            title: Some("OAuth Guide".to_string()),
            file_path: None,
            is_pdf: false,
        },
    ];

    let result = pipeline
        .ingest_documents(&mut db, data_path, "docs", docs)
        .await
        .expect("Failed to ingest");

    assert!(result.chunks_created >= 1, "Should create at least 1 chunk, got {}", result.chunks_created);

    // Verify BM25 index was populated by searching
    let bm25_results = bm25_index.search("JWT authentication", 10).expect("BM25 search failed");
    assert!(!bm25_results.is_empty(), "BM25 should find JWT document");

    let bm25_results = bm25_index.search("OAuth2 authorization", 10).expect("BM25 search failed");
    assert!(!bm25_results.is_empty(), "BM25 should find OAuth document");
}

#[tokio::test]
async fn test_bm25_boosts_exact_keyword_matches() {
    let dir = tempdir().expect("Failed to create temp dir");
    let data_path = dir.path();

    let embedder = Arc::new(Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder"));
    let bm25_index = Arc::new(BM25Index::open(data_path).expect("Failed to create BM25 index"));
    let mut db = VectorDB::new(data_path.to_str().unwrap()).await.expect("Failed to create db");

    let pipeline = IngestPipeline::new(Arc::clone(&embedder), Arc::clone(&bm25_index));

    // Ingest documents - one with exact "JWT" keyword, one semantically related but without exact match
    let docs = vec![
        eywa::DocumentInput {
            content: "JWT tokens are used for stateless authentication in web applications. JSON Web Tokens provide a compact and self-contained way for securely transmitting information. They are commonly used in authorization scenarios and can carry claims about the user.".to_string(),
            title: Some("JWT Guide".to_string()),
            file_path: None,
            is_pdf: false,
        },
        eywa::DocumentInput {
            content: "Token-based authentication provides secure access control mechanisms for modern web applications. This approach eliminates the need for server-side sessions and enables horizontal scaling of backend services.".to_string(),
            title: Some("Auth Overview".to_string()),
            file_path: None,
            is_pdf: false,
        },
    ];

    pipeline
        .ingest_documents(&mut db, data_path, "docs", docs)
        .await
        .expect("Failed to ingest");

    // BM25 search for "JWT" should find the first document
    let bm25_results = bm25_index.search("JWT", 10).expect("BM25 search failed");
    assert!(!bm25_results.is_empty(), "BM25 should find document with JWT keyword");

    // The top result should be the one with exact "JWT" match
    let top_chunk_id = &bm25_results[0].chunk_id;
    let content_store = ContentStore::open(&data_path.join("content.db")).expect("Failed to open content store");
    let contents = content_store.get_chunks(&[top_chunk_id.as_str()]).expect("Failed to get chunk");

    assert!(
        contents[0].1.contains("JWT"),
        "Top BM25 result should contain exact 'JWT' keyword"
    );
}

#[tokio::test]
async fn test_delete_source_removes_from_bm25() {
    let dir = tempdir().expect("Failed to create temp dir");
    let data_path = dir.path();

    let embedder = Arc::new(Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder"));
    let bm25_index = Arc::new(BM25Index::open(data_path).expect("Failed to create BM25 index"));
    let mut db = VectorDB::new(data_path.to_str().unwrap()).await.expect("Failed to create db");

    let pipeline = IngestPipeline::new(Arc::clone(&embedder), Arc::clone(&bm25_index));

    // Ingest documents to two sources (must be long enough to create chunks)
    let docs1 = vec![eywa::DocumentInput {
        content: "GraphQL is a query language for APIs that provides a complete description of the data in your API. It gives clients the power to ask for exactly what they need and nothing more. GraphQL makes it easier to evolve APIs over time.".to_string(),
        title: Some("GraphQL".to_string()),
        file_path: None,
        is_pdf: false,
    }];
    let docs2 = vec![eywa::DocumentInput {
        content: "REST APIs use HTTP methods for CRUD operations on resources. Representational State Transfer is an architectural style that defines constraints for creating web services. REST APIs are stateless and cacheable.".to_string(),
        title: Some("REST".to_string()),
        file_path: None,
        is_pdf: false,
    }];

    pipeline
        .ingest_documents(&mut db, data_path, "source-graphql", docs1)
        .await
        .expect("Failed to ingest");
    pipeline
        .ingest_documents(&mut db, data_path, "source-rest", docs2)
        .await
        .expect("Failed to ingest");

    // Verify both are searchable
    let graphql_results = bm25_index.search("GraphQL", 10).expect("BM25 search failed");
    assert!(!graphql_results.is_empty(), "Should find GraphQL before delete");

    let rest_results = bm25_index.search("REST", 10).expect("BM25 search failed");
    assert!(!rest_results.is_empty(), "Should find REST before delete");

    // Delete one source from BM25
    bm25_index.delete_source("source-graphql").expect("Failed to delete source");

    // Verify deleted source is no longer searchable
    let graphql_results = bm25_index.search("GraphQL", 10).expect("BM25 search failed");
    assert!(graphql_results.is_empty(), "Should NOT find GraphQL after delete");

    // Other source should still be searchable
    let rest_results = bm25_index.search("REST", 10).expect("BM25 search failed");
    assert!(!rest_results.is_empty(), "Should still find REST after delete");
}

#[tokio::test]
async fn test_hybrid_search_combines_vector_and_bm25() {
    let dir = tempdir().expect("Failed to create temp dir");
    let data_path = dir.path();

    let embedder = Arc::new(Embedder::new_with_model(&EmbeddingModel::default(), &DevicePreference::Cpu, false).expect("Failed to create embedder"));
    let bm25_index = Arc::new(BM25Index::open(data_path).expect("Failed to create BM25 index"));
    let mut db = VectorDB::new(data_path.to_str().unwrap()).await.expect("Failed to create db");

    let pipeline = IngestPipeline::new(Arc::clone(&embedder), Arc::clone(&bm25_index));

    // Ingest documents with different characteristics (must be long enough to create chunks):
    // - Doc 1: Has exact keyword "WebSocket" (good for BM25)
    // - Doc 2: Semantically about real-time communication (good for vector)
    let docs = vec![
        eywa::DocumentInput {
            content: "WebSocket provides full-duplex communication channels over a single TCP connection. The WebSocket protocol enables interaction between a web browser or client application and a web server with lower overhead than HTTP polling.".to_string(),
            title: Some("WebSocket Protocol".to_string()),
            file_path: None,
            is_pdf: false,
        },
        eywa::DocumentInput {
            content: "Real-time bidirectional data streaming for interactive applications enables instant updates without page refreshes. This technology powers live chat, notifications, collaborative editing, and gaming applications.".to_string(),
            title: Some("Streaming Guide".to_string()),
            file_path: None,
            is_pdf: false,
        },
    ];

    pipeline
        .ingest_documents(&mut db, data_path, "docs", docs)
        .await
        .expect("Failed to ingest");

    // Vector search for semantic query
    let query = "real-time communication protocol";
    let query_embedding = embedder.embed(query).expect("Failed to embed query");
    let vector_results = db.search(&query_embedding, 10).await.expect("Vector search failed");

    // BM25 search for exact keyword
    let bm25_results = bm25_index.search("WebSocket", 10).expect("BM25 search failed");

    // Both should return results
    assert!(!vector_results.is_empty(), "Vector search should return results");
    assert!(!bm25_results.is_empty(), "BM25 search should return results");

    // Verify we can combine them (simulating hybrid search)
    let vector_ids: Vec<_> = vector_results.iter().map(|r| r.id.clone()).collect();
    let bm25_ids: Vec<_> = bm25_results.iter().map(|r| r.chunk_id.clone()).collect();

    // At least one ID should be in both (the WebSocket doc is semantically relevant too)
    // Or we have complementary results from both systems
    let total_unique: std::collections::HashSet<_> = vector_ids.iter().chain(bm25_ids.iter()).collect();
    assert!(total_unique.len() >= 2, "Hybrid search should cover multiple documents");
}

# ADR-003: Vector Store Selection for Enterprise RAG

## Status

Accepted

## Chinese Summary

`enterprise-rag-engine` v1.0 will use Qdrant as the primary vector store adapter.

The project will keep `BaseVectorStore` as the stable boundary, so Milvus, pgvector,
Elasticsearch, or OpenSearch can be added later without changing parser, splitter,
embedding, retriever, or evaluator code.

## Context

The project is moving from document ingestion into retrieval. The vector store must support:

- dense embedding search for BGE-M3 style vectors;
- metadata filtering by `tenant_id`, `document_id`, `page_number`, and `chunk_type`;
- predictable behavior for multi-tenant enterprise knowledge bases;
- future hybrid retrieval with BM25, dense vectors, sparse vectors, reranking, and RRF;
- local development without forcing a large distributed stack;
- a clear migration path if scale or deployment constraints change.

The current codebase already has:

- `BaseVectorStore`;
- `VectorStoreRecord`;
- `VectorSearchRequest`;
- `VectorStoreFilter`;
- `QdrantVectorStore`;
- metadata filter translation tests;
- deterministic index-parameter benchmark profiles.

## Decision

Use Qdrant as the primary vector store for v1.0.

The default retrieval architecture is:

```text
DocumentChunk
  -> Embedding
  -> VectorStoreRecord
  -> QdrantVectorStore
  -> BaseRetriever
```

Qdrant is selected because it fits the current project shape:

- it is purpose-built for vector search;
- payload indexes are first-class and directly relevant to metadata-filtered RAG;
- the project already has a tested adapter;
- it supports hybrid and multi-stage query patterns, including dense + sparse fusion;
- it is easier to run locally than a full distributed vector stack;
- it leaves room for cloud or self-hosted deployment later.

## Alternatives Considered

### Qdrant

Qdrant combines vector indexes and payload indexes. Its documentation states that vector indexes
speed up vector search while payload indexes speed up filtering. It also recommends creating
payload indexes for fields used in filters, especially before ingesting data so filter-aware HNSW
can benefit from them.

This is a direct match for enterprise RAG, where retrieval often uses semantic similarity plus
tenant, document, page, or chunk-type constraints.

Decision: choose as primary vector store.

### Milvus

Milvus is strong for larger-scale vector workloads. It supports multiple vector index types and
metrics, including FLAT, IVF variants, HNSW, DISKANN, dense vectors, binary vectors, and sparse
vectors. It is a good candidate when the project needs distributed ingestion, larger deployments,
or storage/index flexibility beyond the first v1.0 system.

The tradeoff is operational complexity. For this project stage, the extra distributed-system
surface area is not yet justified.

Decision: keep as future scale-out option.

### pgvector

pgvector is attractive when the team already uses PostgreSQL and wants the smallest operational
footprint. It supports HNSW and IVFFlat indexes, and its HNSW docs describe the usual tradeoff:
better speed-recall behavior than IVFFlat, but slower builds and higher memory use.

The main concern is filtered vector search at RAG scale. pgvector filtering is closely tied to
PostgreSQL query planning, partial indexes, partitioning, and iterative scans. That can be good for
small and medium systems, but it is not the cleanest default for this project, which is intended to
show dedicated retrieval engineering.

Decision: use for lightweight deployments, not as primary v1.0 store.

### Elasticsearch or OpenSearch

Elasticsearch and OpenSearch are strong search platforms, especially when the organization already
runs them for logs, full-text search, observability, or search dashboards. Elasticsearch has
`dense_vector` support and kNN search. OpenSearch positions vector search as a full vector database
solution and supports semantic, hybrid, multimodal, and neural sparse search flows.

For this project, they are better treated as lexical or hybrid-search companions than as the
primary vector abstraction. W6 will add BM25 and hybrid retrieval; at that point OpenSearch or
Elasticsearch can become a strong candidate for the lexical side.

Decision: use as BM25/full-text or hybrid companion, not the primary vector store for v1.0.

## Selection Matrix

| Option | Strength | Weakness | Project role |
|---|---|---|---|
| Qdrant | Vector-native, strong metadata filtering, local-friendly, hybrid-query path | Another service to operate | Primary v1.0 vector store |
| Milvus | Large-scale vector database, many index types, distributed architecture | More operational complexity | Future scale-out option |
| pgvector | Uses PostgreSQL, simple ops, transactional metadata | Filtered ANN depends on SQL planner and index strategy | Lightweight deployment option |
| Elasticsearch | Search platform, dense vector and kNN, strong text search ecosystem | Vector storage/search tied to search-engine model | Full-text and hybrid companion |
| OpenSearch | Open-source search stack, vector, neural and hybrid search flows | Heavier platform surface | Full-text and hybrid companion |

## Implementation Consequences

The project should:

- keep `BaseVectorStore` as the only retrieval persistence contract;
- keep Qdrant-specific code inside `enterprise_rag_engine.retrieval.qdrant`;
- create payload indexes for high-value filter fields before production ingestion;
- continue using fake clients for adapter unit tests;
- add real Qdrant integration tests later behind an optional environment flag;
- use benchmark scripts to compare index profiles before changing defaults;
- avoid leaking Qdrant SDK types into parser, splitter, embedder, or retriever contracts.

Recommended payload index fields:

| Field | Type | Reason |
|---|---|---|
| `tenant_id` | keyword | tenant isolation and SaaS-style filtering |
| `document_id` | keyword | document-scoped retrieval |
| `chunk_type` | keyword | text/table/image/formula filtering |
| `page_number` | integer | page-level retrieval and citation filtering |
| `end_page_number` | integer | page-range chunks |
| `content_hash` | keyword | deduplication and re-indexing diagnostics |

## Migration Policy

If Qdrant no longer fits, add a new adapter rather than rewriting retrieval:

```text
BaseVectorStore
  -> QdrantVectorStore
  -> MilvusVectorStore
  -> PgVectorStore
  -> OpenSearchVectorStore
```

Migration should be driven by evidence:

- dataset size;
- recall@k;
- filtered recall@k;
- P50/P95/P99 latency;
- write throughput;
- index build time;
- memory and disk cost;
- operational burden.

## Sources

- Qdrant indexing and payload index documentation: https://qdrant.tech/documentation/manage-data/indexing/
- Qdrant hybrid query documentation: https://qdrant.tech/documentation/search/hybrid-queries/
- Milvus vector index documentation: https://milvus.io/docs/index-vector-fields.md
- pgvector documentation: https://github.com/pgvector/pgvector
- Elasticsearch dense vector documentation: https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector
- OpenSearch vector search documentation: https://docs.opensearch.org/latest/vector-search/

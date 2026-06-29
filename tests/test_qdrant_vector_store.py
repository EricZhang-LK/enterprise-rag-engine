from dataclasses import dataclass
from typing import Any

import pytest

from enterprise_rag_engine import (
    ChunkMetadata,
    DocumentChunk,
    VectorSearchRequest,
    VectorStoreFilter,
    VectorStoreRecord,
)
from enterprise_rag_engine.retrieval import QdrantVectorStore


@dataclass(frozen=True, slots=True)
class FakeScoredPoint:
    payload: dict[str, Any]
    score: float


class FakeQdrantClient:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[dict[str, Any]]]] = []
        self.queries: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []
        self.query_response: list[FakeScoredPoint] = []

    def upsert(self, *, collection_name: str, points: list[dict[str, Any]]) -> None:
        self.upserts.append((collection_name, points))

    def query_points(
        self,
        *,
        collection_name: str,
        query: tuple[float, ...],
        query_filter: dict[str, Any] | None,
        limit: int,
        with_payload: bool,
    ) -> list[FakeScoredPoint]:
        self.queries.append(
            {
                "collection_name": collection_name,
                "query": query,
                "query_filter": query_filter,
                "limit": limit,
                "with_payload": with_payload,
            }
        )
        return self.query_response

    def delete(self, *, collection_name: str, points_selector: dict[str, Any]) -> None:
        self.deletes.append(
            {
                "collection_name": collection_name,
                "points_selector": points_selector,
            }
        )


def test_qdrant_vector_store_upserts_chunks_with_payload_metadata() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client, collection_name="rag_chunks", vector_size=3)
    chunk = _chunk()

    result = store.upsert((_record(chunk, (0.1, 0.2, 0.3)),))

    collection_name, points = client.upserts[0]
    assert result.affected_count == 1
    assert collection_name == "rag_chunks"
    assert points == [
        {
            "id": "43ecbe12-c355-5c67-afcc-150df92f8680",
            "vector": (0.1, 0.2, 0.3),
            "payload": {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "content": "Qdrant stores dense vectors with structured payload.",
                "chunk_type": "text",
                "parent_id": None,
                "source_uri": "demo.md",
                "page_number": 3,
                "end_page_number": 3,
                "section_path": ["Vector DB"],
                "tenant_id": "tenant-a",
                "content_hash": "hash-1",
                "chunk_index": 0,
                "chunk_count": 1,
                "chunk_role": "standalone",
                "splitter": "recursive",
                "token_count": 8,
                "start_char": 0,
                "end_char": 52,
                "has_table": False,
                "metadata": {"source": "unit-test"},
            },
        }
    ]


def test_qdrant_vector_store_search_restores_retrieval_results_and_filters() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client, collection_name="rag_chunks", vector_size=3)
    chunk = _chunk()
    store.upsert((_record(chunk, (0.1, 0.2, 0.3)),))
    payload = client.upserts[0][1][0]["payload"]
    client.query_response = [FakeScoredPoint(payload=payload, score=0.92)]

    results = store.search(
        VectorSearchRequest(
            query_embedding=(0.3, 0.2, 0.1),
            top_k=5,
            filters=VectorStoreFilter.exact({"tenant_id": "tenant-a", "document_id": "doc-1"}),
        )
    )

    assert client.queries == [
        {
            "collection_name": "rag_chunks",
            "query": (0.3, 0.2, 0.1),
            "query_filter": {
                "must": [
                    {"key": "tenant_id", "match": {"value": "tenant-a"}},
                    {"key": "document_id", "match": {"value": "doc-1"}},
                ]
            },
            "limit": 5,
            "with_payload": True,
        }
    ]
    assert len(results) == 1
    assert results[0].chunk.id == "chunk-1"
    assert results[0].chunk.metadata.tenant_id == "tenant-a"
    assert results[0].score == 0.92
    assert results[0].rank == 1
    assert results[0].retriever == "qdrant"


def test_qdrant_vector_store_delete_removes_points_by_document_id() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client, collection_name="rag_chunks", vector_size=3)

    result = store.delete(VectorStoreFilter.equals("document_id", "doc-1"))

    assert result.affected_count == 0
    assert client.deletes == [
        {
            "collection_name": "rag_chunks",
            "points_selector": {
                "filter": {
                    "must": [{"key": "document_id", "match": {"value": "doc-1"}}]
                }
            },
        }
    ]


def test_qdrant_vector_store_rejects_embedding_shape_mismatches() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client, collection_name="rag_chunks", vector_size=3)

    with pytest.raises(ValueError, match="embedding dimension must be 3"):
        store.upsert((_record(_chunk(), (0.1, 0.2)),))


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        content="Qdrant stores dense vectors with structured payload.",
        metadata=ChunkMetadata(
            source_uri="demo.md",
            document_id="doc-1",
            page_number=3,
            end_page_number=3,
            section_path=("Vector DB",),
            tenant_id="tenant-a",
            content_hash="hash-1",
            chunk_index=0,
            chunk_count=1,
            splitter="recursive",
            token_count=8,
            start_char=0,
            end_char=52,
            metadata={"source": "unit-test"},
        ),
    )


def _record(chunk: DocumentChunk, embedding: tuple[float, ...]) -> VectorStoreRecord:
    return VectorStoreRecord(chunk=chunk, embedding=embedding)

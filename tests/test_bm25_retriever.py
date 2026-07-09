import pytest

from enterprise_rag_engine import ChunkMetadata, ChunkType, DocumentChunk
from enterprise_rag_engine.retrieval import BM25Retriever


def test_bm25_retriever_ranks_chunks_by_lexical_relevance() -> None:
    retriever = BM25Retriever(
        (
            _chunk("chunk-vector", "Qdrant vector payload filter", tenant_id="tenant-a"),
            _chunk(
                "chunk-bm25",
                "BM25 keyword search exact keyword matching",
                tenant_id="tenant-a",
            ),
            _chunk("chunk-dense", "Dense embedding semantic retrieval", tenant_id="tenant-a"),
        )
    )

    results = retriever.retrieve("keyword search", top_k=2)

    assert [result.chunk.id for result in results] == ["chunk-bm25"]
    assert results[0].query == "keyword search"
    assert results[0].rank == 1
    assert results[0].score > 0
    assert results[0].retriever == "bm25"
    assert "keyword" in (results[0].explanation or "")


def test_bm25_retriever_supports_metadata_filters() -> None:
    retriever = BM25Retriever(
        (
            _chunk("tenant-a-hit", "metadata filter search", tenant_id="tenant-a"),
            _chunk("tenant-b-hit", "metadata filter search", tenant_id="tenant-b"),
        )
    )

    results = retriever.retrieve("metadata filter", top_k=5, filters={"tenant_id": "tenant-b"})

    assert [result.chunk.id for result in results] == ["tenant-b-hit"]


def test_bm25_retriever_uses_document_length_normalization() -> None:
    retriever = BM25Retriever(
        (
            _chunk("short", "hybrid search", tenant_id="tenant-a"),
            _chunk(
                "long",
                "hybrid "
                + "background " * 40
                + "search",
                tenant_id="tenant-a",
            ),
        )
    )

    results = retriever.retrieve("hybrid search", top_k=2)

    assert [result.chunk.id for result in results] == ["short", "long"]


def test_bm25_retriever_rejects_invalid_requests() -> None:
    retriever = BM25Retriever((_chunk("chunk-1", "valid content", tenant_id="tenant-a"),))

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(" ", top_k=1)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("valid", top_k=0)


def test_bm25_retriever_returns_empty_tuple_when_no_terms_match() -> None:
    retriever = BM25Retriever((_chunk("chunk-1", "vector database", tenant_id="tenant-a"),))

    assert retriever.retrieve("unrelated", top_k=3) == ()


def _chunk(chunk_id: str, content: str, *, tenant_id: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="doc-1",
        content=content,
        chunk_type=ChunkType.TEXT,
        metadata=ChunkMetadata(
            source_uri="demo.md",
            document_id="doc-1",
            tenant_id=tenant_id,
            section_path=("Retrieval",),
        ),
    )

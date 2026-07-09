from collections.abc import Sequence

import pytest

from enterprise_rag_engine import (
    BaseEmbedder,
    BaseVectorStore,
    ChunkMetadata,
    DocumentChunk,
    EmbeddingBatchResult,
    EmbeddingRequest,
    EmbeddingResult,
    RetrievalResult,
    VectorSearchRequest,
    VectorStoreFilter,
    VectorStoreRecord,
    VectorStoreWriteResult,
)
from enterprise_rag_engine.retrieval import DenseRetriever


class RecordingEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    def embed(self, request: EmbeddingRequest) -> EmbeddingBatchResult:
        self.requests.append(request)
        result = EmbeddingResult(
            text=request.texts[0],
            embedding=(0.1, 0.2, 0.3),
            model=request.model,
        )
        return EmbeddingBatchResult.from_results(results=(result,), model=request.model)


class RecordingVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.search_requests: list[VectorSearchRequest] = []
        self.search_results: tuple[RetrievalResult, ...] = (
            RetrievalResult(
                query="old query",
                chunk=_chunk("chunk-1", tenant_id="tenant-a"),
                score=0.91,
                rank=1,
                retriever="qdrant",
                explanation="vector_score=0.91",
            ),
        )

    def upsert(self, records: Sequence[VectorStoreRecord]) -> VectorStoreWriteResult:
        return VectorStoreWriteResult(affected_count=len(records))

    def search(self, request: VectorSearchRequest) -> tuple[RetrievalResult, ...]:
        self.search_requests.append(request)
        return self.search_results

    def delete(self, filters: VectorStoreFilter) -> VectorStoreWriteResult:
        return VectorStoreWriteResult(affected_count=0)


def test_dense_retriever_embeds_query_and_searches_vector_store() -> None:
    embedder = RecordingEmbedder()
    vector_store = RecordingVectorStore()
    retriever = DenseRetriever(embedder=embedder, vector_store=vector_store, model="fake-dense")

    results = retriever.retrieve(
        "What is vector search?",
        top_k=5,
        filters={"tenant_id": "tenant-a", "document_id": "doc-1"},
    )

    assert embedder.requests == [
        EmbeddingRequest(texts=("What is vector search?",), model="fake-dense", normalize=True)
    ]
    assert vector_store.search_requests == [
        VectorSearchRequest(
            query_embedding=(0.1, 0.2, 0.3),
            top_k=5,
            filters=VectorStoreFilter.exact({"tenant_id": "tenant-a", "document_id": "doc-1"}),
            query_text="What is vector search?",
        )
    ]
    assert len(results) == 1
    assert results[0].query == "What is vector search?"
    assert results[0].chunk.id == "chunk-1"
    assert results[0].score == 0.91
    assert results[0].rank == 1
    assert results[0].retriever == "dense"
    assert results[0].explanation == "vector_score=0.91"


def test_dense_retriever_uses_empty_filter_when_filters_are_missing() -> None:
    embedder = RecordingEmbedder()
    vector_store = RecordingVectorStore()
    retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)

    retriever.retrieve("semantic retrieval", top_k=3)

    assert vector_store.search_requests[0].filters == VectorStoreFilter.empty()


def test_dense_retriever_rejects_blank_query_and_invalid_top_k() -> None:
    retriever = DenseRetriever(
        embedder=RecordingEmbedder(),
        vector_store=RecordingVectorStore(),
    )

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(" ", top_k=3)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("semantic retrieval", top_k=0)


def _chunk(chunk_id: str, *, tenant_id: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="doc-1",
        content="Dense retrieval searches semantically similar chunks.",
        metadata=ChunkMetadata(source_uri="demo.md", document_id="doc-1", tenant_id=tenant_id),
    )

from collections.abc import Sequence

import pytest

from enterprise_rag_engine import (
    BaseReranker,
    BaseRetriever,
    ChunkMetadata,
    DocumentChunk,
    RetrievalResult,
)
from enterprise_rag_engine.retrieval import TwoStageRetriever


class RecordingRetriever(BaseRetriever):
    """Return fixed recall candidates and record the pipeline request."""

    def __init__(self, results: tuple[RetrievalResult, ...]) -> None:
        self._results = results
        self.requests: list[tuple[str, int, dict[str, str] | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        self.requests.append((query, top_k, filters))
        return self._results[:top_k]


class RecordingReranker(BaseReranker):
    """Return selected candidates in reverse order to make re-ranking observable."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, tuple[RetrievalResult, ...], int]] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        selected = tuple(reversed(candidates))[:top_k]
        self.requests.append((query, tuple(candidates), top_k))
        return tuple(
            result.model_copy(update={"rank": rank, "retriever": "fake_reranker"})
            for rank, result in enumerate(selected, start=1)
        )


def test_two_stage_retriever_recalls_candidate_pool_then_reranks_final_results() -> None:
    retriever = RecordingRetriever(
        (
            _result("chunk-1", 0.9, 1),
            _result("chunk-2", 0.8, 2),
            _result("chunk-3", 0.7, 3),
        )
    )
    reranker = RecordingReranker()
    pipeline = TwoStageRetriever(retriever=retriever, reranker=reranker, candidate_top_k=50)

    results = pipeline.retrieve(
        "How does two-stage retrieval work?",
        top_k=2,
        filters={"tenant_id": "tenant-a"},
    )

    assert retriever.requests == [
        (
            "How does two-stage retrieval work?",
            50,
            {"tenant_id": "tenant-a"},
        )
    ]
    assert reranker.requests[0][0] == "How does two-stage retrieval work?"
    assert [item.chunk.id for item in reranker.requests[0][1]] == ["chunk-1", "chunk-2", "chunk-3"]
    assert reranker.requests[0][2] == 2
    assert [item.chunk.id for item in results] == ["chunk-3", "chunk-2"]
    assert [item.rank for item in results] == [1, 2]


def test_two_stage_retriever_uses_configured_candidate_pool_size() -> None:
    retriever = RecordingRetriever((_result("chunk-1", 0.9, 1),))
    reranker = RecordingReranker()
    pipeline = TwoStageRetriever(retriever=retriever, reranker=reranker, candidate_top_k=4)

    pipeline.retrieve("rerank candidates", top_k=2)

    assert retriever.requests == [("rerank candidates", 4, None)]


def test_two_stage_retriever_rejects_invalid_candidate_pool_and_requests() -> None:
    retriever = RecordingRetriever(())
    reranker = RecordingReranker()

    with pytest.raises(ValueError, match="candidate_top_k"):
        TwoStageRetriever(retriever=retriever, reranker=reranker, candidate_top_k=0)

    pipeline = TwoStageRetriever(retriever=retriever, reranker=reranker, candidate_top_k=3)
    with pytest.raises(ValueError, match="query"):
        pipeline.retrieve(" ", top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        pipeline.retrieve("rerank candidates", top_k=0)
    with pytest.raises(ValueError, match="candidate_top_k"):
        pipeline.retrieve("rerank candidates", top_k=4)

    assert retriever.requests == []
    assert reranker.requests == []


def _result(chunk_id: str, score: float, rank: int) -> RetrievalResult:
    return RetrievalResult(
        query="old query",
        chunk=DocumentChunk(
            id=chunk_id,
            document_id="doc-1",
            content=f"content for {chunk_id}",
            metadata=ChunkMetadata(
                source_uri="demo.md",
                document_id="doc-1",
                tenant_id="tenant-a",
            ),
        ),
        score=score,
        rank=rank,
        retriever="rrf",
    )

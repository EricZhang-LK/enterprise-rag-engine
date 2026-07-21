from __future__ import annotations

import threading

import pytest

from enterprise_rag_engine import BaseRetriever, ChunkMetadata, DocumentChunk, RetrievalResult
from enterprise_rag_engine.retrieval import HybridRetriever


class BarrierRetriever(BaseRetriever):
    """Test double that proves both retrieval branches run at the same time."""

    def __init__(
        self,
        *,
        results: tuple[RetrievalResult, ...],
        barrier: threading.Barrier | None = None,
    ) -> None:
        self._results = results
        self._barrier = barrier
        self.requests: list[tuple[str, int, dict[str, str] | None]] = []
        self.thread_ids: list[int] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        self.requests.append((query, top_k, filters))
        self.thread_ids.append(threading.get_ident())
        if self._barrier is not None:
            self._barrier.wait(timeout=1)
        return self._results[:top_k]


def test_hybrid_retriever_runs_dense_and_lexical_recall_in_parallel() -> None:
    barrier = threading.Barrier(2)
    dense = BarrierRetriever(results=(_result("dense-1", "dense", 0.91, 1),), barrier=barrier)
    lexical = BarrierRetriever(
        results=(_result("bm25-1", "bm25", 6.2, 1),),
        barrier=barrier,
    )
    retriever = HybridRetriever(dense_retriever=dense, lexical_retriever=lexical)

    results = retriever.retrieve("hybrid retrieval", top_k=2, filters={"tenant_id": "tenant-a"})

    assert dense.requests == [("hybrid retrieval", 2, {"tenant_id": "tenant-a"})]
    assert lexical.requests == [("hybrid retrieval", 2, {"tenant_id": "tenant-a"})]
    assert dense.thread_ids[0] != threading.get_ident()
    assert lexical.thread_ids[0] != threading.get_ident()
    assert [result.chunk.id for result in results] == ["bm25-1", "dense-1"]


def test_hybrid_retriever_fuses_dense_and_lexical_ranks_with_rrf() -> None:
    dense = BarrierRetriever(
        results=(
            _result("dense-1", "dense", 0.91, 1),
            _result("shared", "dense", 0.82, 2),
        )
    )
    lexical = BarrierRetriever(
        results=(
            _result("shared", "bm25", 8.4, 1),
            _result("bm25-1", "bm25", 5.1, 2),
        )
    )
    retriever = HybridRetriever(dense_retriever=dense, lexical_retriever=lexical)

    results = retriever.retrieve("hybrid retrieval", top_k=3)

    assert [result.chunk.id for result in results] == ["shared", "dense-1", "bm25-1"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.retriever for result in results] == ["rrf", "rrf", "rrf"]
    assert results[0].query == "hybrid retrieval"
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[0].explanation == "rrf_sources=dense#2,bm25#1"


def test_hybrid_retriever_supports_per_branch_candidate_limits() -> None:
    dense = BarrierRetriever(
        results=(
            _result("dense-1", "dense", 0.91, 1),
            _result("dense-2", "dense", 0.82, 2),
            _result("dense-3", "dense", 0.73, 3),
        )
    )
    lexical = BarrierRetriever(
        results=(
            _result("bm25-1", "bm25", 6.2, 1),
            _result("bm25-2", "bm25", 5.1, 2),
            _result("bm25-3", "bm25", 4.0, 3),
        )
    )
    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
        dense_top_k=3,
        lexical_top_k=1,
    )

    results = retriever.retrieve("hybrid retrieval", top_k=2)

    assert dense.requests == [("hybrid retrieval", 3, None)]
    assert lexical.requests == [("hybrid retrieval", 1, None)]
    assert [result.chunk.id for result in results] == ["bm25-1", "dense-1"]


def test_hybrid_retriever_rejects_invalid_requests_and_candidate_limits() -> None:
    dense = BarrierRetriever(results=())
    lexical = BarrierRetriever(results=())

    with pytest.raises(ValueError, match="dense_top_k"):
        HybridRetriever(dense_retriever=dense, lexical_retriever=lexical, dense_top_k=0)

    retriever = HybridRetriever(dense_retriever=dense, lexical_retriever=lexical)
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(" ", top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("hybrid retrieval", top_k=0)


def _result(chunk_id: str, retriever: str, score: float, rank: int) -> RetrievalResult:
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
        retriever=retriever,
    )

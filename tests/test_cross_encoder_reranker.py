from collections.abc import Sequence
from typing import Any

import pytest

from enterprise_rag_engine import ChunkMetadata, DocumentChunk, RetrievalResult
from enterprise_rag_engine.retrieval import BGE_RERANKER_V2_M3_MODEL_NAME, CrossEncoderReranker


class FakeCrossEncoder:
    """Record pair inputs while returning deterministic cross-encoder logits."""

    def __init__(self, scores: Any) -> None:
        self._scores = scores
        self.calls: list[dict[str, Any]] = []

    def predict(self, pairs: Sequence[tuple[str, str]], *, batch_size: int) -> Any:
        self.calls.append({"pairs": tuple(pairs), "batch_size": batch_size})
        return self._scores


def test_cross_encoder_reranker_rescores_and_reranks_retrieval_candidates() -> None:
    model = FakeCrossEncoder(scores=(-2.0, 1.0, 0.0))
    reranker = CrossEncoderReranker(model=model, batch_size=8)

    results = reranker.rerank(
        "How does RRF work?",
        (
            _result("chunk-a", "first candidate", 0.9, 1),
            _result("chunk-b", "best candidate", 0.8, 2),
            _result("chunk-c", "middle candidate", 0.7, 3),
        ),
        top_k=2,
    )

    assert reranker.model_name == BGE_RERANKER_V2_M3_MODEL_NAME
    assert model.calls == [
        {
            "pairs": (
                ("How does RRF work?", "first candidate"),
                ("How does RRF work?", "best candidate"),
                ("How does RRF work?", "middle candidate"),
            ),
            "batch_size": 8,
        }
    ]
    assert [result.chunk.id for result in results] == ["chunk-b", "chunk-c"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score == pytest.approx(0.7310585786)
    assert results[0].retriever == "cross_encoder"
    assert results[0].explanation == "cross_encoder_raw_score=1.0"


def test_cross_encoder_reranker_uses_chunk_id_to_break_equal_score_ties() -> None:
    reranker = CrossEncoderReranker(model=FakeCrossEncoder(scores=(0.5, 0.5)))

    results = reranker.rerank(
        "rerank candidates",
        (_result("chunk-z", "z", 0.9, 1), _result("chunk-a", "a", 0.8, 2)),
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["chunk-a", "chunk-z"]


def test_cross_encoder_reranker_returns_empty_tuple_without_calling_the_model() -> None:
    model = FakeCrossEncoder(scores=())
    reranker = CrossEncoderReranker(model=model)

    assert reranker.rerank("rerank candidates", (), top_k=3) == ()
    assert model.calls == []


def test_cross_encoder_reranker_rejects_invalid_requests_and_model_output() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderReranker(model=FakeCrossEncoder(scores=()), batch_size=0)

    reranker = CrossEncoderReranker(model=FakeCrossEncoder(scores=(0.5,)))
    candidates = (_result("chunk-a", "candidate", 0.9, 1), _result("chunk-b", "other", 0.8, 2))

    with pytest.raises(ValueError, match="query"):
        reranker.rerank(" ", candidates, top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("rerank candidates", candidates, top_k=0)
    with pytest.raises(ValueError, match="same number"):
        reranker.rerank("rerank candidates", candidates, top_k=2)


def _result(chunk_id: str, content: str, score: float, rank: int) -> RetrievalResult:
    return RetrievalResult(
        query="old query",
        chunk=DocumentChunk(
            id=chunk_id,
            document_id="doc-1",
            content=content,
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

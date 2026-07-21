import pytest

from enterprise_rag_engine import ChunkMetadata, DocumentChunk, RetrievalResult
from enterprise_rag_engine.retrieval import RRFusion


def test_rrf_fusion_rewards_chunks_retrieved_by_multiple_channels() -> None:
    fusion = RRFusion(rank_constant=60)
    dense_results = (
        _result("dense-only", "dense", 0.91, 1),
        _result("shared", "dense", 0.82, 2),
    )
    lexical_results = (
        _result("shared", "bm25", 8.4, 1),
        _result("bm25-only", "bm25", 5.1, 2),
    )

    results = fusion.fuse(
        query="hybrid retrieval",
        result_groups=(dense_results, lexical_results),
        top_k=3,
    )

    assert [result.chunk.id for result in results] == ["shared", "dense-only", "bm25-only"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[0].retriever == "rrf"
    assert results[0].query == "hybrid retrieval"
    assert results[0].explanation == "rrf_sources=dense#2,bm25#1"


def test_rrf_fusion_counts_a_chunk_once_per_retrieval_channel() -> None:
    fusion = RRFusion()
    dense_results = (
        _result("shared", "dense", 0.91, 1),
        _result("shared", "dense", 0.82, 2),
    )
    lexical_results = (_result("lexical-only", "bm25", 8.4, 1),)

    results = fusion.fuse(
        query="hybrid retrieval",
        result_groups=(dense_results, lexical_results),
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["lexical-only", "shared"]
    assert results[1].score == pytest.approx(1 / 61)


def test_rrf_fusion_uses_chunk_id_to_break_equal_score_ties() -> None:
    fusion = RRFusion()

    results = fusion.fuse(
        query="hybrid retrieval",
        result_groups=(
            (_result("chunk-z", "dense", 0.91, 1),),
            (_result("chunk-a", "bm25", 8.4, 1),),
        ),
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["chunk-a", "chunk-z"]


def test_rrf_fusion_rejects_invalid_configuration_and_requests() -> None:
    with pytest.raises(ValueError, match="rank_constant"):
        RRFusion(rank_constant=0)

    fusion = RRFusion()
    with pytest.raises(ValueError, match="query"):
        fusion.fuse(query=" ", result_groups=(), top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        fusion.fuse(query="hybrid retrieval", result_groups=(), top_k=0)


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

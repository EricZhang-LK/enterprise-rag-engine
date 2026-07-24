from math import log2

import pytest

from enterprise_rag_engine.evals.retrieval import (
    RetrievalEvaluationCase,
    evaluate_retrieval_rankings,
)


def test_retrieval_metrics_are_perfect_when_all_relevant_chunks_rank_first() -> None:
    cases = (
        RetrievalEvaluationCase(
            id="case-1",
            query="what is RRF",
            relevant_chunk_ids=frozenset({"chunk-rrf"}),
        ),
        RetrievalEvaluationCase(
            id="case-2",
            query="what is reranking",
            relevant_chunk_ids=frozenset({"chunk-rerank"}),
        ),
    )

    metrics = evaluate_retrieval_rankings(
        cases=cases,
        rankings_by_case={
            "case-1": ("chunk-rrf", "noise"),
            "case-2": ("chunk-rerank", "noise"),
        },
    )

    assert metrics.case_count == 2
    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg_at_10 == 1.0


def test_retrieval_metrics_measure_partial_recall_and_late_relevant_ranks() -> None:
    cases = (
        RetrievalEvaluationCase(
            id="case-1",
            query="hybrid retrieval",
            relevant_chunk_ids=frozenset({"chunk-a", "chunk-b"}),
        ),
        RetrievalEvaluationCase(
            id="case-2",
            query="cross encoder",
            relevant_chunk_ids=frozenset({"chunk-c"}),
        ),
    )

    metrics = evaluate_retrieval_rankings(
        cases=cases,
        rankings_by_case={
            "case-1": ("noise", "chunk-a", "noise-2"),
            "case-2": ("noise", "noise-2", "noise-3", "noise-4", "noise-5", "chunk-c"),
        },
    )

    assert metrics.recall_at_5 == pytest.approx(0.25)
    assert metrics.recall_at_10 == pytest.approx(0.75)
    assert metrics.mrr == pytest.approx((1 / 2 + 1 / 6) / 2)
    assert metrics.ndcg_at_10 == pytest.approx(
        ((1 / log2(3)) / (1 + 1 / log2(3)) + 1 / log2(7)) / 2
    )


def test_retrieval_metrics_reject_duplicate_cases_and_missing_rankings() -> None:
    duplicate_cases = (
        RetrievalEvaluationCase(
            id="case-1",
            query="RRF",
            relevant_chunk_ids=frozenset({"chunk-rrf"}),
        ),
        RetrievalEvaluationCase(
            id="case-1",
            query="Rerank",
            relevant_chunk_ids=frozenset({"chunk-rerank"}),
        ),
    )

    with pytest.raises(ValueError, match="unique"):
        evaluate_retrieval_rankings(cases=duplicate_cases, rankings_by_case={})

    case = RetrievalEvaluationCase(
        id="case-2",
        query="Hybrid",
        relevant_chunk_ids=frozenset({"chunk-hybrid"}),
    )
    with pytest.raises(ValueError, match="missing rankings"):
        evaluate_retrieval_rankings(cases=(case,), rankings_by_case={})

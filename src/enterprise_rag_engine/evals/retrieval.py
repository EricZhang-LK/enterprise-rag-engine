from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import log2


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One query with the chunk identifiers accepted as relevant evidence."""

    id: str
    query: str
    relevant_chunk_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.id.strip():
            msg = "case id must not be blank"
            raise ValueError(msg)
        if not self.query.strip():
            msg = "query must not be blank"
            raise ValueError(msg)
        if not self.relevant_chunk_ids:
            msg = "relevant_chunk_ids must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Aggregate retrieval quality metrics using binary relevance judgments."""

    case_count: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float

    def as_dict(self) -> dict[str, int | float]:
        """Return a serializable metric mapping for scripts and reports."""

        return asdict(self)


def evaluate_retrieval_rankings(
    *,
    cases: Sequence[RetrievalEvaluationCase],
    rankings_by_case: Mapping[str, Sequence[str]],
) -> RetrievalMetrics:
    """Calculate Recall@5, Recall@10, MRR, and NDCG@10 from ranked chunk ids."""

    if not cases:
        msg = "at least one evaluation case is required"
        raise ValueError(msg)
    case_ids = tuple(case.id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        msg = "evaluation case ids must be unique"
        raise ValueError(msg)
    missing_case_ids = set(case_ids) - set(rankings_by_case)
    if missing_case_ids:
        msg = "missing rankings for evaluation cases: " + ", ".join(sorted(missing_case_ids))
        raise ValueError(msg)

    recalls_at_5: list[float] = []
    recalls_at_10: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs_at_10: list[float] = []
    for case in cases:
        ranking = tuple(rankings_by_case[case.id])
        recalls_at_5.append(_recall_at_k(ranking, case.relevant_chunk_ids, k=5))
        recalls_at_10.append(_recall_at_k(ranking, case.relevant_chunk_ids, k=10))
        reciprocal_ranks.append(_reciprocal_rank(ranking, case.relevant_chunk_ids))
        ndcgs_at_10.append(_ndcg_at_k(ranking, case.relevant_chunk_ids, k=10))

    case_count = len(cases)
    return RetrievalMetrics(
        case_count=case_count,
        recall_at_5=sum(recalls_at_5) / case_count,
        recall_at_10=sum(recalls_at_10) / case_count,
        mrr=sum(reciprocal_ranks) / case_count,
        ndcg_at_10=sum(ndcgs_at_10) / case_count,
    )


def _recall_at_k(ranking: Sequence[str], relevant_chunk_ids: frozenset[str], *, k: int) -> float:
    retrieved_relevant = set(ranking[:k]) & relevant_chunk_ids
    return len(retrieved_relevant) / len(relevant_chunk_ids)


def _reciprocal_rank(ranking: Sequence[str], relevant_chunk_ids: frozenset[str]) -> float:
    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1 / rank
    return 0.0


def _ndcg_at_k(ranking: Sequence[str], relevant_chunk_ids: frozenset[str], *, k: int) -> float:
    dcg = sum(
        1 / log2(rank + 1)
        for rank, chunk_id in enumerate(ranking[:k], start=1)
        if chunk_id in relevant_chunk_ids
    )
    ideal_result_count = min(k, len(relevant_chunk_ids))
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_result_count + 1))
    return dcg / ideal_dcg

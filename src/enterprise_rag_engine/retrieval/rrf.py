from __future__ import annotations

from dataclasses import dataclass, field

from enterprise_rag_engine.models import RetrievalResult


@dataclass(slots=True)
class _FusedCandidate:
    result: RetrievalResult
    score: float = 0.0
    sources: list[str] = field(default_factory=list)


class RRFusion:
    """Fuse ranked retrieval channels with Reciprocal Rank Fusion."""

    def __init__(self, *, rank_constant: int = 60) -> None:
        if rank_constant < 1:
            msg = "rank_constant must be greater than 0"
            raise ValueError(msg)
        self._rank_constant = rank_constant

    def fuse(
        self,
        *,
        query: str,
        result_groups: tuple[tuple[RetrievalResult, ...], ...],
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        """Combine ranked result groups without comparing their original scores."""

        if top_k < 1:
            msg = "top_k must be greater than 0"
            raise ValueError(msg)
        if not query.strip():
            msg = "query must not be blank"
            raise ValueError(msg)

        candidates: dict[str, _FusedCandidate] = {}
        for results in result_groups:
            seen_chunk_ids: set[str] = set()
            for result in results:
                chunk_id = result.chunk.id
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                candidate = candidates.setdefault(chunk_id, _FusedCandidate(result=result))
                candidate.score += 1 / (self._rank_constant + result.rank)
                candidate.sources.append(f"{result.retriever}#{result.rank}")

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.score, candidate.result.chunk.id),
        )
        return tuple(
            candidate.result.model_copy(
                update={
                    "query": query,
                    "score": candidate.score,
                    "rank": rank,
                    "retriever": "rrf",
                    "explanation": "rrf_sources=" + ",".join(candidate.sources),
                }
            )
            for rank, candidate in enumerate(ranked[:top_k], start=1)
        )

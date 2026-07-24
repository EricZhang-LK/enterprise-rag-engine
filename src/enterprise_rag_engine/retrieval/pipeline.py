from __future__ import annotations

from enterprise_rag_engine.interfaces import BaseReranker, BaseRetriever
from enterprise_rag_engine.models import RetrievalResult


class TwoStageRetriever(BaseRetriever):
    """Recall a broad candidate pool, then re-rank a smaller final result set."""

    def __init__(
        self,
        *,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        candidate_top_k: int = 50,
    ) -> None:
        if candidate_top_k < 1:
            msg = "candidate_top_k must be greater than 0"
            raise ValueError(msg)
        self._retriever = retriever
        self._reranker = reranker
        self._candidate_top_k = candidate_top_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Recall candidates first, then return the highest-ranked re-ranked results."""

        if top_k < 1:
            msg = "top_k must be greater than 0"
            raise ValueError(msg)
        if top_k > self._candidate_top_k:
            msg = "top_k must not be greater than candidate_top_k"
            raise ValueError(msg)
        if not query.strip():
            msg = "query must not be blank"
            raise ValueError(msg)

        candidates = self._retriever.retrieve(
            query,
            top_k=self._candidate_top_k,
            filters=filters,
        )
        return self._reranker.rerank(query, candidates, top_k=top_k)

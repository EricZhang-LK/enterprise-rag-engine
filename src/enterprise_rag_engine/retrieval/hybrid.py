from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from enterprise_rag_engine.interfaces import BaseRetriever
from enterprise_rag_engine.models import RetrievalResult
from enterprise_rag_engine.retrieval.rrf import RRFusion


class HybridRetriever(BaseRetriever):
    """Run dense and lexical recall concurrently, then fuse candidates with RRF."""

    def __init__(
        self,
        *,
        dense_retriever: BaseRetriever,
        lexical_retriever: BaseRetriever,
        dense_top_k: int | None = None,
        lexical_top_k: int | None = None,
        fusion: RRFusion | None = None,
    ) -> None:
        _validate_candidate_limit("dense_top_k", dense_top_k)
        _validate_candidate_limit("lexical_top_k", lexical_top_k)
        self._dense_retriever = dense_retriever
        self._lexical_retriever = lexical_retriever
        self._dense_top_k = dense_top_k
        self._lexical_top_k = lexical_top_k
        self._fusion = fusion or RRFusion()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Collect dense and BM25 candidates in parallel, then fuse their ranks."""

        if top_k < 1:
            msg = "top_k must be greater than 0"
            raise ValueError(msg)
        if not query.strip():
            msg = "query must not be blank"
            raise ValueError(msg)

        dense_limit = self._dense_top_k or top_k
        lexical_limit = self._lexical_top_k or top_k
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval") as executor:
            dense_future = executor.submit(
                self._dense_retriever.retrieve,
                query,
                top_k=dense_limit,
                filters=filters,
            )
            lexical_future = executor.submit(
                self._lexical_retriever.retrieve,
                query,
                top_k=lexical_limit,
                filters=filters,
            )
            dense_results = dense_future.result()
            lexical_results = lexical_future.result()

        return self._fusion.fuse(
            query=query,
            top_k=top_k,
            result_groups=(dense_results, lexical_results),
        )


def _validate_candidate_limit(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        msg = f"{name} must be greater than 0"
        raise ValueError(msg)

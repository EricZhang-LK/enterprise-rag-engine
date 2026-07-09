from enterprise_rag_engine.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore
from enterprise_rag_engine.models import (
    EmbeddingRequest,
    FilterValue,
    RetrievalResult,
    VectorSearchRequest,
    VectorStoreFilter,
)


class DenseRetriever(BaseRetriever):
    """Retriever that embeds natural-language queries before vector search."""

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        model: str | None = None,
        normalize: bool = True,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._model = model
        self._normalize = normalize

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Embed a query, search the vector store, and normalize result metadata."""

        if top_k < 1:
            msg = "top_k must be greater than 0"
            raise ValueError(msg)
        if not query.strip():
            msg = "query must not be blank"
            raise ValueError(msg)

        embedding_batch = self._embedder.embed(
            EmbeddingRequest(texts=(query,), model=self._model, normalize=self._normalize)
        )
        vector_results = self._vector_store.search(
            VectorSearchRequest(
                query_embedding=embedding_batch.embeddings[0],
                top_k=top_k,
                filters=_vector_filter(filters),
                query_text=query,
            )
        )
        return tuple(
            result.model_copy(update={"query": query, "retriever": "dense"})
            for result in vector_results
        )


def _vector_filter(filters: dict[str, str] | None) -> VectorStoreFilter:
    if not filters:
        return VectorStoreFilter.empty()
    conditions: dict[str, FilterValue] = dict(filters)
    return VectorStoreFilter.exact(conditions)

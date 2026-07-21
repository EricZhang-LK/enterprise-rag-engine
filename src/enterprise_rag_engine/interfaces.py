import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence

from enterprise_rag_engine.models import (
    Document,
    DocumentChunk,
    Embedding,
    EmbeddingBatchResult,
    EmbeddingRequest,
    OCRResult,
    ParseResult,
    RetrievalResult,
    VectorSearchRequest,
    VectorStoreFilter,
    VectorStoreRecord,
    VectorStoreWriteResult,
)


class BaseParser(ABC):
    """Parse a source into a normalized document and optional chunks."""

    @abstractmethod
    def parse(self, source_uri: str) -> ParseResult:
        """Parse a document from a local path, URL, or storage URI."""


class BaseSplitter(ABC):
    """Split normalized documents into retrieval-ready chunks."""

    @abstractmethod
    def split(self, document: Document) -> tuple[DocumentChunk, ...]:
        """Split one document into chunks while preserving metadata."""


class BaseEmbedder(ABC):
    """Convert text inputs into embedding vectors."""

    @abstractmethod
    def embed(self, request: EmbeddingRequest) -> EmbeddingBatchResult:
        """Embed a batch of text inputs and return rich per-text results."""

    async def aembed(self, request: EmbeddingRequest) -> EmbeddingBatchResult:
        """Embed texts asynchronously using a thread wrapper by default."""

        return await asyncio.to_thread(self.embed, request)

    def embed_texts(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Compatibility helper for callers that only need raw vectors."""

        return self.embed(EmbeddingRequest(texts=tuple(texts))).embeddings


class BaseVectorStore(ABC):
    """Persist and search chunk embeddings."""

    @abstractmethod
    def upsert(self, records: Sequence[VectorStoreRecord]) -> VectorStoreWriteResult:
        """Insert or update chunk vectors and return the write result."""

    @abstractmethod
    def search(self, request: VectorSearchRequest) -> tuple[RetrievalResult, ...]:
        """Search for the most relevant chunks."""

    @abstractmethod
    def delete(self, filters: VectorStoreFilter) -> VectorStoreWriteResult:
        """Delete vectors matching a filter."""


class BaseRetriever(ABC):
    """Retrieve relevant chunks for a natural-language query."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Return ranked retrieval results for a query."""


class BaseReranker(ABC):
    """Re-score retrieved candidates for one query and return a new ranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        """Return the highest-scoring candidates after precision-oriented re-ranking."""


class BaseEvaluator(ABC):
    """Evaluate a component or pipeline and return numeric metrics."""

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Return metric names and values."""


class BaseOCRProvider(ABC):
    """Extract text from image-like document pages."""

    @abstractmethod
    def extract_text(self, source_uri: str) -> tuple[OCRResult, ...]:
        """Return OCR results for a source document or image."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from enterprise_rag_engine.models import (
    Document,
    DocumentChunk,
    OCRResult,
    ParseResult,
    RetrievalResult,
)

Embedding = tuple[float, ...]


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
    def embed_texts(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed a batch of text inputs."""


class BaseVectorStore(ABC):
    """Persist and search chunk embeddings."""

    @abstractmethod
    def upsert(self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Embedding]) -> None:
        """Insert or update chunk vectors."""

    @abstractmethod
    def search(
        self,
        query_embedding: Embedding,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Search for the most relevant chunks."""

    @abstractmethod
    def delete(self, document_id: str) -> None:
        """Delete all vectors associated with one document."""


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

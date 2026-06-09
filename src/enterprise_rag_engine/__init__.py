"""Enterprise RAG Engine package."""

from enterprise_rag_engine.document_pipeline.parsers import (
    DocxParser,
    MarkdownParser,
    PdfTextParser,
    StructuredPdfParser,
)
from enterprise_rag_engine.interfaces import (
    BaseEmbedder,
    BaseEvaluator,
    BaseParser,
    BaseRetriever,
    BaseSplitter,
    BaseVectorStore,
    Embedding,
)
from enterprise_rag_engine.models import (
    ChunkMetadata,
    ChunkType,
    Document,
    DocumentChunk,
    DocumentType,
    ParseResult,
    ParseStatus,
    RetrievalResult,
)

__version__ = "0.1.0"

__all__ = [
    "BaseEmbedder",
    "BaseEvaluator",
    "BaseParser",
    "BaseRetriever",
    "BaseSplitter",
    "BaseVectorStore",
    "ChunkMetadata",
    "ChunkType",
    "Document",
    "DocumentChunk",
    "DocumentType",
    "DocxParser",
    "Embedding",
    "MarkdownParser",
    "ParseResult",
    "ParseStatus",
    "PdfTextParser",
    "RetrievalResult",
    "StructuredPdfParser",
    "__version__",
]

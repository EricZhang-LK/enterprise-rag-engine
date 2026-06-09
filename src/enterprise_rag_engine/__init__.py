"""Enterprise RAG Engine package."""

from enterprise_rag_engine.document_pipeline.parsers import (
    DocxParser,
    MarkdownParser,
    OcrDocumentParser,
    PdfTextParser,
    StructuredPdfParser,
)
from enterprise_rag_engine.interfaces import (
    BaseEmbedder,
    BaseEvaluator,
    BaseOCRProvider,
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
    OCRResult,
    OcrStatus,
    ParseResult,
    ParseStatus,
    RetrievalResult,
    TableBlock,
)

__version__ = "0.1.0"

__all__ = [
    "BaseEmbedder",
    "BaseEvaluator",
    "BaseOCRProvider",
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
    "OCRResult",
    "OcrDocumentParser",
    "OcrStatus",
    "ParseResult",
    "ParseStatus",
    "PdfTextParser",
    "RetrievalResult",
    "StructuredPdfParser",
    "TableBlock",
    "__version__",
]

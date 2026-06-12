"""Enterprise RAG Engine package."""

from enterprise_rag_engine.document_pipeline.parsers import (
    DocxParser,
    MarkdownParser,
    OcrDocumentParser,
    PdfTextParser,
    StructuredPdfParser,
)
from enterprise_rag_engine.document_pipeline.splitters import (
    ParentChildSplitter,
    RecursiveSplitter,
    SemanticSplitter,
)
from enterprise_rag_engine.document_pipeline.tokenization import (
    BaseTokenCounter,
    HuggingFaceTokenCounter,
    TiktokenTokenCounter,
    TokenBudget,
    TokenCounter,
    TokenSpan,
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
    ChunkRole,
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
    "BaseTokenCounter",
    "BaseVectorStore",
    "ChunkMetadata",
    "ChunkRole",
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
    "ParentChildSplitter",
    "PdfTextParser",
    "RecursiveSplitter",
    "RetrievalResult",
    "SemanticSplitter",
    "StructuredPdfParser",
    "TableBlock",
    "HuggingFaceTokenCounter",
    "TokenBudget",
    "TokenCounter",
    "TokenSpan",
    "TiktokenTokenCounter",
    "__version__",
]

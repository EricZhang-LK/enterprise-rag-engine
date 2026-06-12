from collections.abc import Callable
from hashlib import sha256
from importlib import import_module
from time import perf_counter
from typing import Any

from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentType,
    ParseResult,
    ParseStatus,
)

DoclingConverterFactory = Callable[[], Any]


class StructuredPdfParser(BaseParser):
    """Parse PDFs into structured Markdown using Docling when available."""

    def __init__(self, converter_factory: DoclingConverterFactory | None = None) -> None:
        self._converter_factory = converter_factory or _default_docling_converter

    def parse(self, source_uri: str) -> ParseResult:
        started_at = perf_counter()
        try:
            converter = self._converter_factory()
            result = converter.convert(source_uri)
            markdown = str(result.document.export_to_markdown()).strip()
        except Exception as exc:
            document = Document(source_uri=source_uri, type=DocumentType.PDF, content="")
            return ParseResult(
                document=document,
                status=ParseStatus.FAILED,
                errors=(f"Failed to parse structured PDF: {exc}",),
                elapsed_ms=_elapsed_ms(started_at),
            )

        if not markdown:
            document = Document(source_uri=source_uri, type=DocumentType.PDF, content="")
            return ParseResult(
                document=document,
                status=ParseStatus.FAILED,
                errors=("No structured Markdown exported from PDF.",),
                elapsed_ms=_elapsed_ms(started_at),
            )

        document = Document(
            source_uri=source_uri,
            type=DocumentType.PDF,
            content=markdown,
            metadata={
                "parser": self.__class__.__name__,
                "structured_format": "markdown",
                "backend": "docling",
            },
        )
        chunk = DocumentChunk(
            document_id=document.id,
            content=markdown,
            metadata=ChunkMetadata(
                source_uri=source_uri,
                document_id=document.id,
                content_hash=sha256(markdown.encode("utf-8")).hexdigest(),
                chunk_index=0,
                chunk_count=1,
                splitter="StructuredPdfParser",
                start_char=0,
                end_char=len(markdown),
                metadata={"structured_format": "markdown", "backend": "docling"},
            ),
            start_char=0,
            end_char=len(markdown),
        )
        return ParseResult(
            document=document,
            chunks=(chunk,),
            status=ParseStatus.SUCCEEDED,
            elapsed_ms=_elapsed_ms(started_at),
        )


def _default_docling_converter() -> Any:
    try:
        module = import_module("docling.document_converter")
    except ImportError as exc:
        raise RuntimeError(
            "Docling is not installed. Install with `pip install -e .[structured]`."
        ) from exc
    return module.DocumentConverter()


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)

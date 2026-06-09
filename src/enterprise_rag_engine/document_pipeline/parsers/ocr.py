from hashlib import sha256
from time import perf_counter

from enterprise_rag_engine.interfaces import BaseOCRProvider, BaseParser
from enterprise_rag_engine.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentType,
    OCRResult,
    OcrStatus,
    ParseResult,
    ParseStatus,
)


class OcrDocumentParser(BaseParser):
    """Turn OCR provider results into the common ParseResult contract."""

    def __init__(self, provider: BaseOCRProvider) -> None:
        self._provider = provider

    def parse(self, source_uri: str) -> ParseResult:
        started_at = perf_counter()
        try:
            ocr_results = self._provider.extract_text(source_uri)
        except Exception as exc:
            document = Document(source_uri=source_uri, type=DocumentType.UNKNOWN, content="")
            return ParseResult(
                document=document,
                status=ParseStatus.FAILED,
                errors=(f"OCR provider failed: {exc}",),
                elapsed_ms=_elapsed_ms(started_at),
            )

        errors = tuple(error for result in ocr_results for error in result.errors)
        successful_results = tuple(
            result for result in ocr_results if result.status is OcrStatus.SUCCEEDED
        )
        content = "\n\n".join(result.text for result in successful_results if result.text)

        document = Document(
            source_uri=source_uri,
            type=DocumentType.UNKNOWN,
            content=content,
            metadata={
                "parser": self.__class__.__name__,
                "ocr_page_count": len(ocr_results),
            },
        )
        chunks = tuple(
            _ocr_chunk(source_uri=source_uri, document_id=document.id, result=result)
            for result in successful_results
            if result.text
        )

        return ParseResult(
            document=document,
            chunks=chunks,
            status=_status_for(content=content, errors=errors, total_results=len(ocr_results)),
            errors=errors,
            elapsed_ms=_elapsed_ms(started_at),
        )


def _ocr_chunk(*, source_uri: str, document_id: str, result: OCRResult) -> DocumentChunk:
    return DocumentChunk(
        document_id=document_id,
        content=result.text,
        metadata=ChunkMetadata(
            source_uri=source_uri,
            document_id=document_id,
            page_number=result.page_number,
            content_hash=sha256(result.text.encode("utf-8")).hexdigest(),
        ),
    )


def _status_for(*, content: str, errors: tuple[str, ...], total_results: int) -> ParseStatus:
    if content and not errors:
        return ParseStatus.SUCCEEDED
    if content and errors:
        return ParseStatus.PARTIAL
    if total_results == 0:
        return ParseStatus.FAILED
    return ParseStatus.FAILED


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)

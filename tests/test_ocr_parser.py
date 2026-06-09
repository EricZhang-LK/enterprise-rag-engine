from enterprise_rag_engine import (
    BaseOCRProvider,
    OcrDocumentParser,
    OCRResult,
    OcrStatus,
    ParseStatus,
)


class FakeOCRProvider(BaseOCRProvider):
    def extract_text(self, source_uri: str) -> tuple[OCRResult, ...]:
        return (
            OCRResult(
                text="page one text",
                status=OcrStatus.SUCCEEDED,
                page_number=1,
                confidence=0.95,
            ),
            OCRResult(
                text="",
                status=OcrStatus.FAILED,
                page_number=2,
                errors=("page 2 low contrast",),
            ),
        )


class EmptyOCRProvider(BaseOCRProvider):
    def extract_text(self, source_uri: str) -> tuple[OCRResult, ...]:
        return ()


class FailingOCRProvider(BaseOCRProvider):
    def extract_text(self, source_uri: str) -> tuple[OCRResult, ...]:
        raise RuntimeError("ocr backend unavailable")


def test_ocr_document_parser_returns_partial_result_with_page_metadata() -> None:
    result = OcrDocumentParser(FakeOCRProvider()).parse("scan.pdf")

    assert result.status is ParseStatus.PARTIAL
    assert result.document.content == "page one text"
    assert result.document.metadata["ocr_page_count"] == 2
    assert result.chunk_count == 1
    assert result.chunks[0].metadata.page_number == 1
    assert result.errors == ("page 2 low contrast",)


def test_ocr_document_parser_marks_empty_provider_result_as_failed() -> None:
    result = OcrDocumentParser(EmptyOCRProvider()).parse("empty-scan.pdf")

    assert result.status is ParseStatus.FAILED
    assert result.document.content == ""
    assert result.chunk_count == 0


def test_ocr_document_parser_handles_provider_exception() -> None:
    result = OcrDocumentParser(FailingOCRProvider()).parse("bad-scan.pdf")

    assert result.status is ParseStatus.FAILED
    assert result.errors == ("OCR provider failed: ocr backend unavailable",)

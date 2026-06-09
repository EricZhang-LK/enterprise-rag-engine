from typing import Any

from enterprise_rag_engine import DocumentType, ParseStatus, PdfTextParser


class FakePdfMetadata:
    title = "Demo PDF"
    author = "Test Author"
    creator = None
    producer = "Unit Test"
    subject = None


class FakePdfPage:
    def __init__(self, text: str | None, should_fail: bool = False) -> None:
        self._text = text
        self._should_fail = should_fail

    def extract_text(self) -> str | None:
        if self._should_fail:
            raise ValueError("broken page")
        return self._text


class FakePdfReader:
    metadata = FakePdfMetadata()

    def __init__(self, _source_uri: str) -> None:
        self.pages = [
            FakePdfPage(" page one \n\n paragraph "),
            FakePdfPage("page two"),
        ]


class FakePartialPdfReader:
    metadata = None

    def __init__(self, _source_uri: str) -> None:
        self.pages = [
            FakePdfPage("page one"),
            FakePdfPage(None, should_fail=True),
        ]


class FakeScannedPdfReader:
    metadata = None

    def __init__(self, _source_uri: str) -> None:
        self.pages = [FakePdfPage("")]


def failing_reader(_source_uri: str) -> Any:
    raise ValueError("not a pdf")


def test_pdf_text_parser_extracts_document_and_page_chunks() -> None:
    parser = PdfTextParser(reader_factory=FakePdfReader)

    result = parser.parse("demo.pdf")

    assert result.status is ParseStatus.SUCCEEDED
    assert result.document.type is DocumentType.PDF
    assert result.document.title == "Demo PDF"
    assert result.document.metadata["page_count"] == 2
    assert result.document.metadata["author"] == "Test Author"
    assert result.document.content == "page one\nparagraph\n\npage two"
    assert result.chunk_count == 2
    assert result.chunks[0].metadata.page_number == 1
    assert result.chunks[1].metadata.page_number == 2
    assert result.chunks[0].metadata.content_hash is not None


def test_pdf_text_parser_returns_partial_status_when_one_page_fails() -> None:
    parser = PdfTextParser(reader_factory=FakePartialPdfReader)

    result = parser.parse("partial.pdf")

    assert result.status is ParseStatus.PARTIAL
    assert result.document.content == "page one"
    assert result.chunk_count == 1
    assert result.errors == ("Failed to extract text from page 2: broken page",)


def test_pdf_text_parser_marks_scanned_pdf_without_text_as_failed() -> None:
    parser = PdfTextParser(reader_factory=FakeScannedPdfReader)

    result = parser.parse("scan.pdf")

    assert result.status is ParseStatus.FAILED
    assert result.document.content == ""
    assert result.chunk_count == 0
    assert result.errors == ("No extractable text found. The PDF may require OCR.",)


def test_pdf_text_parser_returns_failed_status_when_reader_cannot_open_pdf() -> None:
    parser = PdfTextParser(reader_factory=failing_reader)

    result = parser.parse("bad.pdf")

    assert result.status is ParseStatus.FAILED
    assert result.document.source_uri == "bad.pdf"
    assert result.errors == ("Failed to read PDF: not a pdf",)

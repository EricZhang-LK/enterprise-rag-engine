from typing import Any

from enterprise_rag_engine import ParseStatus, StructuredPdfParser


class FakeDoclingDocument:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown

    def export_to_markdown(self) -> str:
        return self._markdown


class FakeDoclingResult:
    def __init__(self, markdown: str) -> None:
        self.document = FakeDoclingDocument(markdown)


class FakeDoclingConverter:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown

    def convert(self, _source_uri: str) -> FakeDoclingResult:
        return FakeDoclingResult(self._markdown)


def converter_factory(markdown: str) -> Any:
    return lambda: FakeDoclingConverter(markdown)


def failing_converter_factory() -> Any:
    raise RuntimeError("docling backend failed")


def test_structured_pdf_parser_exports_markdown_chunk() -> None:
    parser = StructuredPdfParser(converter_factory=converter_factory("# Title\n\n| A | B |"))

    result = parser.parse("structured.pdf")

    assert result.status is ParseStatus.SUCCEEDED
    assert result.document.metadata["backend"] == "docling"
    assert result.document.metadata["structured_format"] == "markdown"
    assert result.document.content == "# Title\n\n| A | B |"
    assert result.chunk_count == 1
    assert result.chunks[0].metadata.content_hash is not None


def test_structured_pdf_parser_marks_empty_markdown_as_failed() -> None:
    parser = StructuredPdfParser(converter_factory=converter_factory(""))

    result = parser.parse("empty.pdf")

    assert result.status is ParseStatus.FAILED
    assert result.errors == ("No structured Markdown exported from PDF.",)


def test_structured_pdf_parser_returns_failed_status_when_backend_fails() -> None:
    parser = StructuredPdfParser(converter_factory=failing_converter_factory)

    result = parser.parse("bad.pdf")

    assert result.status is ParseStatus.FAILED
    assert result.errors == ("Failed to parse structured PDF: docling backend failed",)

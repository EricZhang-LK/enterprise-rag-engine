from typing import Any

from enterprise_rag_engine import DocumentType, DocxParser, ParseStatus


class FakeStyle:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeParagraph:
    def __init__(self, text: str, style_name: str) -> None:
        self.text = text
        self.style = FakeStyle(style_name)


class FakeDocxDocument:
    def __init__(self, _source_uri: str) -> None:
        self.paragraphs = [
            FakeParagraph("Product Guide", "Heading 1"),
            FakeParagraph("Intro paragraph.", "Normal"),
            FakeParagraph("Install", "Heading 2"),
            FakeParagraph("Install steps.", "Normal"),
        ]


class FakeEmptyDocxDocument:
    def __init__(self, _source_uri: str) -> None:
        self.paragraphs: list[FakeParagraph] = []


def failing_reader(_source_uri: str) -> Any:
    raise ValueError("not a docx")


def test_docx_parser_preserves_heading_paths() -> None:
    result = DocxParser(reader_factory=FakeDocxDocument).parse("demo.docx")

    assert result.status is ParseStatus.SUCCEEDED
    assert result.document.type is DocumentType.DOCX
    assert result.document.title == "Product Guide"
    assert result.document.metadata["paragraph_count"] == 4
    assert result.chunk_count == 4
    assert result.chunks[0].metadata.section_path == ("Product Guide",)
    assert result.chunks[1].metadata.section_path == ("Product Guide",)
    assert result.chunks[2].metadata.section_path == ("Product Guide", "Install")
    assert result.chunks[3].metadata.section_path == ("Product Guide", "Install")


def test_docx_parser_marks_empty_document_as_failed() -> None:
    result = DocxParser(reader_factory=FakeEmptyDocxDocument).parse("empty.docx")

    assert result.status is ParseStatus.FAILED
    assert result.document.content == ""
    assert result.chunk_count == 0
    assert result.errors == ("No extractable text found in Docx.",)


def test_docx_parser_returns_failed_status_when_reader_cannot_open_docx() -> None:
    result = DocxParser(reader_factory=failing_reader).parse("bad.docx")

    assert result.status is ParseStatus.FAILED
    assert result.document.source_uri == "bad.docx"
    assert result.errors == ("Failed to read Docx: not a docx",)

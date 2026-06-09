from enterprise_rag_engine import DocumentType, MarkdownParser, ParseStatus


def test_markdown_parser_preserves_heading_paths() -> None:
    parser = MarkdownParser(
        text_reader=lambda _source_uri: (
            "# Product Guide\n\n"
            "Intro paragraph.\n\n"
            "## Install\n\n"
            "Install steps.\n\n"
            "### Windows\n\n"
            "Windows steps.\n"
        )
    )

    result = parser.parse("demo.md")

    assert result.status is ParseStatus.SUCCEEDED
    assert result.document.type is DocumentType.MARKDOWN
    assert result.document.title == "Product Guide"
    assert result.chunk_count == 3
    assert result.chunks[0].metadata.section_path == ("Product Guide",)
    assert result.chunks[1].metadata.section_path == ("Product Guide", "Install")
    assert result.chunks[2].metadata.section_path == ("Product Guide", "Install", "Windows")
    assert result.chunks[0].metadata.content_hash is not None


def test_markdown_parser_returns_failed_status_when_file_is_missing() -> None:
    result = MarkdownParser().parse("missing.md")

    assert result.status is ParseStatus.FAILED
    assert result.document.content == ""
    assert result.errors[0].startswith("Failed to read Markdown:")

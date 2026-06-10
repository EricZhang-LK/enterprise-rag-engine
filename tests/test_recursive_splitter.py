import pytest

from enterprise_rag_engine import Document, DocumentType, RecursiveSplitter


def test_recursive_splitter_keeps_short_document_as_single_chunk() -> None:
    document = Document(
        source_uri="demo.md",
        type=DocumentType.MARKDOWN,
        content="# Title\n\nA short paragraph.",
    )

    chunks = RecursiveSplitter(max_chars=200, overlap_chars=20).split(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content
    assert chunks[0].metadata.source_uri == "demo.md"
    assert chunks[0].metadata.document_id == document.id
    assert chunks[0].metadata.content_hash is not None
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(document.content)


def test_recursive_splitter_prefers_paragraph_boundaries() -> None:
    document = Document(
        source_uri="demo.txt",
        type=DocumentType.TEXT,
        content="alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota",
    )

    chunks = RecursiveSplitter(max_chars=25, overlap_chars=0).split(document)

    assert [chunk.content for chunk in chunks] == [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
    ]


def test_recursive_splitter_falls_back_to_character_windows() -> None:
    document = Document(source_uri="demo.txt", type=DocumentType.TEXT, content="x" * 25)

    chunks = RecursiveSplitter(max_chars=10, overlap_chars=0).split(document)

    assert [chunk.content for chunk in chunks] == ["x" * 10, "x" * 10, "x" * 5]
    assert [chunk.start_char for chunk in chunks] == [0, 10, 20]
    assert [chunk.end_char for chunk in chunks] == [10, 20, 25]


def test_recursive_splitter_can_overlap_adjacent_chunks() -> None:
    document = Document(
        source_uri="demo.txt",
        type=DocumentType.TEXT,
        content="alpha beta gamma delta epsilon zeta eta theta",
    )

    chunks = RecursiveSplitter(max_chars=30, overlap_chars=5).split(document)

    assert len(chunks) > 1
    assert chunks[1].start_char is not None
    assert chunks[0].end_char is not None
    assert chunks[1].start_char <= chunks[0].end_char
    assert all(chunk.character_count <= 30 for chunk in chunks)


def test_recursive_splitter_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="overlap_chars must be smaller"):
        RecursiveSplitter(max_chars=10, overlap_chars=10)

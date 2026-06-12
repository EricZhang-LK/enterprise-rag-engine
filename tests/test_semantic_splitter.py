import pytest

from enterprise_rag_engine import Document, DocumentType, SemanticSplitter


def test_semantic_splitter_groups_related_sentences() -> None:
    document = Document(
        source_uri="demo.txt",
        type=DocumentType.TEXT,
        content=(
            "RAG retrieval improves search quality. "
            "RAG retrieval depends on chunk quality. "
            "FastAPI exposes backend APIs."
        ),
    )

    chunks = SemanticSplitter(
        max_chars=120,
        min_chars=20,
        similarity_threshold=0.1,
    ).split(document)

    assert len(chunks) == 2
    assert "chunk quality" in chunks[0].content
    assert chunks[1].content == "FastAPI exposes backend APIs."


def test_semantic_splitter_preserves_original_offsets() -> None:
    document = Document(
        source_uri="demo.txt",
        type=DocumentType.TEXT,
        content="  RAG retrieval improves search.\n\nFastAPI exposes backend APIs.",
    )

    chunks = SemanticSplitter(max_chars=80, min_chars=10).split(document)

    assert chunks[0].start_char == 2
    assert chunks[0].metadata.splitter == "SemanticSplitter"
    assert chunks[0].metadata.start_char == chunks[0].start_char
    assert chunks[0].metadata.end_char == chunks[0].end_char
    assert chunks[0].metadata.token_count is not None
    assert document.content[chunks[0].start_char : chunks[0].end_char] == chunks[0].content
    assert chunks[0].metadata.content_hash is not None


def test_semantic_splitter_falls_back_when_semantic_group_is_too_long() -> None:
    document = Document(
        source_uri="demo.txt",
        type=DocumentType.TEXT,
        content="x" * 35,
    )

    chunks = SemanticSplitter(max_chars=10, min_chars=1, overlap_chars=0).split(document)

    assert [chunk.content for chunk in chunks] == ["x" * 10, "x" * 10, "x" * 10, "x" * 5]
    assert [chunk.start_char for chunk in chunks] == [0, 10, 20, 30]


def test_semantic_splitter_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        SemanticSplitter(similarity_threshold=1.5)

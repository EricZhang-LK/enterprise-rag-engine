import pytest

from enterprise_rag_engine import Document, DocumentType, ParentChildSplitter


def test_parent_child_splitter_returns_parents_and_children() -> None:
    document = Document(
        source_uri="demo.txt",
        type=DocumentType.TEXT,
        content=(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa.\n\n"
            "lambda mu nu xi omicron pi rho sigma tau upsilon."
        ),
    )

    chunks = ParentChildSplitter(
        parent_max_chars=70,
        parent_overlap_chars=0,
        child_max_chars=30,
        child_overlap_chars=0,
    ).split(document)

    parents = [chunk for chunk in chunks if chunk.parent_id is None]
    children = [chunk for chunk in chunks if chunk.parent_id is not None]

    assert len(parents) == 2
    assert len(children) > len(parents)
    assert {child.parent_id for child in children}.issubset({parent.id for parent in parents})


def test_parent_child_splitter_rebases_child_offsets_to_document() -> None:
    document = Document(
        source_uri="demo.txt",
        type=DocumentType.TEXT,
        content="first parent paragraph.\n\nsecond parent paragraph with child chunks.",
    )

    chunks = ParentChildSplitter(
        parent_max_chars=28,
        parent_overlap_chars=0,
        child_max_chars=16,
        child_overlap_chars=0,
    ).split(document)
    second_parent = [chunk for chunk in chunks if chunk.parent_id is None][1]
    second_parent_children = [chunk for chunk in chunks if chunk.parent_id == second_parent.id]

    assert second_parent.start_char is not None
    assert second_parent_children[0].start_char == second_parent.start_char
    assert document.content[
        second_parent_children[0].start_char : second_parent_children[0].end_char
    ] == second_parent_children[0].content


def test_parent_child_splitter_preserves_document_metadata_on_children() -> None:
    document = Document(
        source_uri="demo.txt",
        type=DocumentType.TEXT,
        content="alpha beta gamma delta epsilon zeta eta theta iota kappa",
    )

    chunks = ParentChildSplitter(
        parent_max_chars=40,
        parent_overlap_chars=0,
        child_max_chars=20,
        child_overlap_chars=0,
    ).split(document)
    children = [chunk for chunk in chunks if chunk.parent_id is not None]

    assert children
    assert all(child.document_id == document.id for child in children)
    assert all(child.metadata.source_uri == document.source_uri for child in children)
    assert all(child.metadata.content_hash is not None for child in children)


def test_parent_child_splitter_rejects_child_larger_than_parent() -> None:
    with pytest.raises(ValueError, match="child_max_chars"):
        ParentChildSplitter(parent_max_chars=100, child_max_chars=100)

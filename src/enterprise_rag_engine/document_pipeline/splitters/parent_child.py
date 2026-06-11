from hashlib import sha256

from enterprise_rag_engine.document_pipeline.splitters.recursive import RecursiveSplitter
from enterprise_rag_engine.interfaces import BaseSplitter
from enterprise_rag_engine.models import (
    ChunkMetadata,
    ChunkType,
    Document,
    DocumentChunk,
    DocumentType,
)


class ParentChildSplitter(BaseSplitter):
    """Create large parent chunks and smaller retrieval-oriented child chunks."""

    def __init__(
        self,
        *,
        parent_max_chars: int = 1_600,
        parent_overlap_chars: int = 160,
        child_max_chars: int = 400,
        child_overlap_chars: int = 80,
    ) -> None:
        if child_max_chars >= parent_max_chars:
            msg = "child_max_chars must be smaller than parent_max_chars"
            raise ValueError(msg)

        self.parent_splitter = RecursiveSplitter(
            max_chars=parent_max_chars,
            overlap_chars=parent_overlap_chars,
        )
        self.child_splitter = RecursiveSplitter(
            max_chars=child_max_chars,
            overlap_chars=child_overlap_chars,
        )

    def split(self, document: Document) -> tuple[DocumentChunk, ...]:
        """Return parent chunks followed by their children.

        Parent chunks are stored in the same tuple so a vector index can choose to
        index only children while a document store can still persist parents.
        """

        parent_chunks = self.parent_splitter.split(document)
        result: list[DocumentChunk] = []

        for parent in parent_chunks:
            result.append(parent)
            result.extend(self._child_chunks(document=document, parent=parent))

        return tuple(result)

    def _child_chunks(
        self,
        *,
        document: Document,
        parent: DocumentChunk,
    ) -> tuple[DocumentChunk, ...]:
        parent_start = parent.start_char or 0
        parent_document = Document(
            id=document.id,
            source_uri=document.source_uri,
            type=DocumentType.TEXT,
            title=document.title,
            content=parent.content,
            metadata=document.metadata,
            created_at=document.created_at,
        )
        relative_children = self.child_splitter.split(parent_document)

        return tuple(
            _rebase_child_chunk(
                document=document,
                parent=parent,
                child=child,
                parent_start=parent_start,
            )
            for child in relative_children
        )


def _rebase_child_chunk(
    *,
    document: Document,
    parent: DocumentChunk,
    child: DocumentChunk,
    parent_start: int,
) -> DocumentChunk:
    child_start = None if child.start_char is None else parent_start + child.start_char
    child_end = None if child.end_char is None else parent_start + child.end_char
    metadata = ChunkMetadata(
        source_uri=document.source_uri,
        document_id=document.id,
        page_number=parent.metadata.page_number,
        section_path=parent.metadata.section_path,
        tenant_id=parent.metadata.tenant_id,
        content_hash=_content_hash(child.content),
    )
    return DocumentChunk(
        document_id=document.id,
        content=child.content,
        chunk_type=ChunkType.TEXT,
        metadata=metadata,
        parent_id=parent.id,
        start_char=child_start,
        end_char=child_end,
    )


def _content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()

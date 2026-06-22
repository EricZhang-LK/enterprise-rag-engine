import asyncio
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from enterprise_rag_engine.api.repositories.tasks import (
    InMemoryTaskRepository,
    ParseTask,
)
from enterprise_rag_engine.document_pipeline import AsyncDocumentPipeline, CacheManager
from enterprise_rag_engine.document_pipeline.parsers import (
    DocxParser,
    MarkdownParser,
    PdfTextParser,
)
from enterprise_rag_engine.exceptions import ResourceNotFoundError, ValidationFailedError
from enterprise_rag_engine.interfaces import BaseParser

SUPPORTED_EXTENSIONS = frozenset({".docx", ".markdown", ".md", ".pdf"})


class DocumentStorage(Protocol):
    async def save(self, *, task_id: str, filename: str, content: bytes) -> str: ...


class ParserProvider(Protocol):
    def get(self, source_uri: str) -> BaseParser: ...


class LocalDocumentStorage:
    """Persist uploaded bytes under a server-controlled directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def save(self, *, task_id: str, filename: str, content: bytes) -> str:
        safe_name = Path(filename).name
        destination = self._root / f"{task_id}-{safe_name}"
        await asyncio.to_thread(_write_file, destination, content)
        return str(destination.resolve())


class ParserRegistry:
    """Select a parser by the validated document extension."""

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {
            ".docx": DocxParser(),
            ".markdown": MarkdownParser(),
            ".md": MarkdownParser(),
            ".pdf": PdfTextParser(),
        }

    def get(self, source_uri: str) -> BaseParser:
        extension = Path(source_uri).suffix.lower()
        parser = self._parsers.get(extension)
        if parser is None:
            raise ValidationFailedError(f"Unsupported document type: {extension or 'unknown'}")
        return parser


class DocumentTaskService:
    """Coordinate upload persistence, background parsing, and task queries."""

    def __init__(
        self,
        *,
        repository: InMemoryTaskRepository,
        storage: DocumentStorage,
        parser_registry: ParserProvider,
        max_upload_bytes: int,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._parser_registry = parser_registry
        self._max_upload_bytes = max_upload_bytes
        self._cache = CacheManager()

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    async def submit(
        self,
        *,
        filename: str | None,
        content_type: str | None,
        content: bytes,
    ) -> ParseTask:
        safe_name = _validate_upload(
            filename=filename,
            content=content,
            max_upload_bytes=self._max_upload_bytes,
        )
        task_id = str(uuid4())
        source_uri = await self._storage.save(
            task_id=task_id,
            filename=safe_name,
            content=content,
        )
        return self._repository.create(
            task_id=task_id,
            filename=safe_name,
            source_uri=source_uri,
            content_type=content_type,
        )

    async def process(self, task_id: str) -> None:
        task = self.get(task_id)
        parser = self._parser_registry.get(task.source_uri)
        pipeline = AsyncDocumentPipeline(
            parser,
            cache_manager=self._cache,
            progress_handler=self._repository.apply_event,
        )
        result = await pipeline.parse(task.source_uri, task_id=task.id)
        self._repository.complete(task.id, result)

    def get(self, task_id: str) -> ParseTask:
        task = self._repository.get(task_id)
        if task is None:
            raise ResourceNotFoundError(f"Parse task not found: {task_id}")
        return task


def _validate_upload(*, filename: str | None, content: bytes, max_upload_bytes: int) -> str:
    if filename is None or not filename.strip():
        raise ValidationFailedError("Uploaded document must have a filename")
    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValidationFailedError(f"Unsupported document type. Supported: {supported}")
    if not content:
        raise ValidationFailedError("Uploaded document must not be empty")
    if len(content) > max_upload_bytes:
        raise ValidationFailedError(f"Uploaded document exceeds {max_upload_bytes} bytes")
    return safe_name


def _write_file(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)

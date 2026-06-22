import asyncio
from collections.abc import Awaitable, Callable, Sequence
from inspect import isawaitable
from time import perf_counter
from uuid import uuid4

from enterprise_rag_engine.document_pipeline.cache import CacheManager
from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import (
    Document,
    DocumentType,
    ParseProgressEvent,
    ParseProgressStage,
    ParseResult,
    ParseStatus,
)

QueueItem = tuple[int, str, str]
ParseProgressHandler = Callable[[ParseProgressEvent], Awaitable[None] | None]


class AsyncDocumentPipeline:
    """Async wrapper around synchronous document parsers.

    Most document parsers are CPU-bound or blocking I/O adapters. The pipeline runs
    them in a worker thread so FastAPI handlers can await parsing without blocking
    the event loop.
    """

    def __init__(
        self,
        parser: BaseParser,
        *,
        max_concurrency: int = 4,
        max_queue_size: int = 100,
        cache_manager: CacheManager | None = None,
        progress_handler: ParseProgressHandler | None = None,
    ) -> None:
        if max_concurrency < 1:
            msg = "max_concurrency must be greater than 0"
            raise ValueError(msg)
        if max_queue_size < 1:
            msg = "max_queue_size must be greater than 0"
            raise ValueError(msg)
        self._parser = parser
        self._max_concurrency = max_concurrency
        self._max_queue_size = max_queue_size
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._cache_manager = cache_manager
        self._progress_handler = progress_handler

    async def parse(self, source_uri: str, *, task_id: str | None = None) -> ParseResult:
        """Parse one source without blocking the event loop."""

        task_id = task_id or _new_task_id()
        await self._emit_progress(
            task_id=task_id,
            source_uri=source_uri,
            stage=ParseProgressStage.STARTED,
            progress=0.1,
            message="Document parsing started.",
        )
        cached_result = self._cache_manager.get(source_uri) if self._cache_manager else None
        if cached_result is not None:
            await self._emit_progress(
                task_id=task_id,
                source_uri=source_uri,
                stage=ParseProgressStage.CACHE_HIT,
                progress=1.0,
                status=cached_result.status,
                elapsed_ms=cached_result.elapsed_ms,
                message="Document parse result loaded from cache.",
                metadata={"chunk_count": cached_result.chunk_count},
            )
            return cached_result

        started_at = perf_counter()
        try:
            async with self._semaphore:
                result = await asyncio.to_thread(self._parser.parse, source_uri)
                if self._cache_manager is not None:
                    self._cache_manager.put(source_uri, result)
                await self._emit_progress(
                    task_id=task_id,
                    source_uri=source_uri,
                    stage=_stage_from_status(result.status),
                    progress=1.0,
                    status=result.status,
                    elapsed_ms=result.elapsed_ms,
                    message="Document parsing finished.",
                    metadata={"chunk_count": result.chunk_count},
                )
                return result
        except Exception as exc:
            document = Document(source_uri=source_uri, type=DocumentType.UNKNOWN, content="")
            elapsed_ms = _elapsed_ms(started_at)
            await self._emit_progress(
                task_id=task_id,
                source_uri=source_uri,
                stage=ParseProgressStage.FAILED,
                progress=1.0,
                status=ParseStatus.FAILED,
                elapsed_ms=elapsed_ms,
                message=f"Async parse failed: {exc}",
            )
            return ParseResult(
                document=document,
                status=ParseStatus.FAILED,
                errors=(f"Async parse failed: {exc}",),
                elapsed_ms=elapsed_ms,
            )

    async def parse_many(self, source_uris: Sequence[str]) -> tuple[ParseResult, ...]:
        """Parse a batch with bounded workers and queue backpressure."""

        if not source_uris:
            return ()

        queue: asyncio.Queue[QueueItem | None] = asyncio.Queue(maxsize=self._max_queue_size)
        results: list[ParseResult | None] = [None] * len(source_uris)
        worker_count = min(self._max_concurrency, len(source_uris))
        workers = tuple(
            asyncio.create_task(self._worker(queue=queue, results=results))
            for _ in range(worker_count)
        )

        for index, source_uri in enumerate(source_uris):
            task_id = _new_task_id()
            await self._emit_progress(
                task_id=task_id,
                source_uri=source_uri,
                stage=ParseProgressStage.QUEUED,
                progress=0.0,
                message="Document parse task queued.",
                metadata={"queue_index": index},
            )
            await queue.put((index, source_uri, task_id))

        for _ in workers:
            await queue.put(None)

        await queue.join()
        await asyncio.gather(*workers)
        return tuple(result for result in results if result is not None)

    async def _worker(
        self,
        *,
        queue: asyncio.Queue[QueueItem | None],
        results: list[ParseResult | None],
    ) -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                index, source_uri, task_id = item
                results[index] = await self.parse(source_uri, task_id=task_id)
            finally:
                queue.task_done()

    async def _emit_progress(
        self,
        *,
        task_id: str,
        source_uri: str,
        stage: ParseProgressStage,
        progress: float,
        status: ParseStatus | None = None,
        elapsed_ms: float | None = None,
        message: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Emit one structured progress event for task APIs or SSE adapters."""

        if self._progress_handler is None:
            return
        event = ParseProgressEvent(
            task_id=task_id,
            source_uri=source_uri,
            stage=stage,
            progress=progress,
            status=status,
            elapsed_ms=elapsed_ms,
            message=message,
            metadata=metadata or {},
        )
        maybe_awaitable = self._progress_handler(event)
        if isawaitable(maybe_awaitable):
            await maybe_awaitable


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _new_task_id() -> str:
    return str(uuid4())


def _stage_from_status(status: ParseStatus) -> ParseProgressStage:
    if status is ParseStatus.FAILED:
        return ParseProgressStage.FAILED
    return ParseProgressStage.SUCCEEDED

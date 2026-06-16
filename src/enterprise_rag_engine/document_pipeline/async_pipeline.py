import asyncio
from collections.abc import Sequence
from time import perf_counter

from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import Document, DocumentType, ParseResult, ParseStatus

QueueItem = tuple[int, str]


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

    async def parse(self, source_uri: str) -> ParseResult:
        """Parse one source without blocking the event loop."""

        started_at = perf_counter()
        try:
            async with self._semaphore:
                return await asyncio.to_thread(self._parser.parse, source_uri)
        except Exception as exc:
            document = Document(source_uri=source_uri, type=DocumentType.UNKNOWN, content="")
            return ParseResult(
                document=document,
                status=ParseStatus.FAILED,
                errors=(f"Async parse failed: {exc}",),
                elapsed_ms=_elapsed_ms(started_at),
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
            await queue.put((index, source_uri))

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
                index, source_uri = item
                results[index] = await self.parse(source_uri)
            finally:
                queue.task_done()


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)

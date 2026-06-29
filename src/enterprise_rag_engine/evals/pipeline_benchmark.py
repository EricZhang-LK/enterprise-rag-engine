import tracemalloc
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil
from time import perf_counter

from enterprise_rag_engine.document_pipeline import AsyncDocumentPipeline
from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import ParseProgressEvent, ParseProgressStage, ParseStatus

TERMINAL_STAGES = frozenset(
    {
        ParseProgressStage.CACHE_HIT,
        ParseProgressStage.SUCCEEDED,
        ParseProgressStage.FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkMetrics:
    file_count: int
    succeeded_count: int
    failed_count: int
    max_concurrency: int
    max_queue_size: int
    total_elapsed_ms: float
    throughput_files_per_second: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    peak_memory_mb: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


async def benchmark_pipeline(
    parser: BaseParser,
    source_uris: Sequence[str],
    *,
    max_concurrency: int = 4,
    max_queue_size: int = 8,
) -> PipelineBenchmarkMetrics:
    """Measure bounded batch parsing from queue admission to terminal event."""

    queued_at: dict[str, float] = {}
    latencies_ms: list[float] = []

    def collect_event(event: ParseProgressEvent) -> None:
        now = perf_counter()
        if event.stage is ParseProgressStage.QUEUED:
            queued_at[event.task_id] = now
            return
        if event.stage in TERMINAL_STAGES:
            started_at = queued_at.get(event.task_id)
            if started_at is not None:
                latencies_ms.append((now - started_at) * 1000)

    pipeline = AsyncDocumentPipeline(
        parser,
        max_concurrency=max_concurrency,
        max_queue_size=max_queue_size,
        progress_handler=collect_event,
    )

    tracemalloc.start()
    started_at = perf_counter()
    try:
        results = await pipeline.parse_many(source_uris)
        total_elapsed_seconds = perf_counter() - started_at
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    succeeded_count = sum(result.status is not ParseStatus.FAILED for result in results)
    total_elapsed_ms = total_elapsed_seconds * 1000
    return PipelineBenchmarkMetrics(
        file_count=len(source_uris),
        succeeded_count=succeeded_count,
        failed_count=len(results) - succeeded_count,
        max_concurrency=max_concurrency,
        max_queue_size=max_queue_size,
        total_elapsed_ms=round(total_elapsed_ms, 3),
        throughput_files_per_second=round(
            len(results) / total_elapsed_seconds if total_elapsed_seconds else 0.0,
            3,
        ),
        p50_latency_ms=_percentile(latencies_ms, 0.50),
        p95_latency_ms=_percentile(latencies_ms, 0.95),
        max_latency_ms=round(max(latencies_ms, default=0.0), 3),
        peak_memory_mb=round(peak_memory_bytes / (1024 * 1024), 3),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return the nearest-rank percentile for a latency sample."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)

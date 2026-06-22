from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock

from enterprise_rag_engine.models import ParseProgressEvent, ParseProgressStage, ParseResult


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ParseTask:
    """Current application-level state for one uploaded document."""

    id: str
    filename: str
    source_uri: str
    content_type: str | None
    status: TaskStatus
    stage: ParseProgressStage
    progress: float
    message: str | None
    result: ParseResult | None
    created_at: datetime
    updated_at: datetime


class InMemoryTaskRepository:
    """Thread-safe task repository used by the local v0.4 application."""

    def __init__(self) -> None:
        self._tasks: dict[str, ParseTask] = {}
        self._lock = Lock()

    def create(
        self,
        *,
        task_id: str,
        filename: str,
        source_uri: str,
        content_type: str | None,
    ) -> ParseTask:
        now = datetime.now(UTC)
        task = ParseTask(
            id=task_id,
            filename=filename,
            source_uri=source_uri,
            content_type=content_type,
            status=TaskStatus.QUEUED,
            stage=ParseProgressStage.QUEUED,
            progress=0.0,
            message="Document parse task queued.",
            result=None,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> ParseTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def apply_event(self, event: ParseProgressEvent) -> None:
        with self._lock:
            current = self._tasks.get(event.task_id)
            if current is None:
                return
            self._tasks[event.task_id] = replace(
                current,
                status=_task_status(event.stage),
                stage=event.stage,
                progress=event.progress,
                message=event.message,
                updated_at=event.created_at,
            )

    def complete(self, task_id: str, result: ParseResult) -> ParseTask | None:
        with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                return None
            task = replace(current, result=result, updated_at=datetime.now(UTC))
            self._tasks[task_id] = task
            return task


def _task_status(stage: ParseProgressStage) -> TaskStatus:
    if stage in {ParseProgressStage.QUEUED}:
        return TaskStatus.QUEUED
    if stage in {ParseProgressStage.STARTED}:
        return TaskStatus.RUNNING
    if stage is ParseProgressStage.FAILED:
        return TaskStatus.FAILED
    return TaskStatus.SUCCEEDED

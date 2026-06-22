from datetime import datetime

from pydantic import BaseModel, Field

from enterprise_rag_engine.api.repositories.tasks import TaskStatus
from enterprise_rag_engine.models import ParseProgressStage, ParseStatus


class UploadDocumentResponse(BaseModel):
    task_id: str
    filename: str
    status: TaskStatus
    task_url: str
    created_at: datetime


class ParseTaskResponse(BaseModel):
    task_id: str
    filename: str
    status: TaskStatus
    stage: ParseProgressStage
    progress: float = Field(ge=0, le=1)
    message: str | None
    parse_status: ParseStatus | None
    chunk_count: int | None
    elapsed_ms: float | None
    errors: tuple[str, ...]
    created_at: datetime
    updated_at: datetime

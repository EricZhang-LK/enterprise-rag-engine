from typing import Annotated, cast

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile, status

from enterprise_rag_engine.api.repositories.tasks import ParseTask
from enterprise_rag_engine.api.schemas.tasks import ParseTaskResponse, UploadDocumentResponse
from enterprise_rag_engine.api.services.documents import DocumentTaskService

router = APIRouter()


def get_document_task_service(request: Request) -> DocumentTaskService:
    return cast(DocumentTaskService, request.app.state.document_task_service)


@router.post(
    "/documents/upload",
    response_model=UploadDocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["documents"],
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
) -> UploadDocumentResponse:
    service = get_document_task_service(request)
    # Reading one extra byte lets validation reject oversized uploads without
    # buffering an unbounded request body in application memory.
    content = await file.read(service.max_upload_bytes + 1)
    task = await service.submit(
        filename=file.filename,
        content_type=file.content_type,
        content=content,
    )
    background_tasks.add_task(service.process, task.id)
    return UploadDocumentResponse(
        task_id=task.id,
        filename=task.filename,
        status=task.status,
        task_url=str(request.url_for("get_parse_task", task_id=task.id)),
        created_at=task.created_at,
    )


@router.get(
    "/tasks/{task_id}",
    response_model=ParseTaskResponse,
    tags=["tasks"],
    name="get_parse_task",
)
def get_parse_task(request: Request, task_id: str) -> ParseTaskResponse:
    task = get_document_task_service(request).get(task_id)
    return _task_response(task)


def _task_response(task: ParseTask) -> ParseTaskResponse:
    result = task.result
    return ParseTaskResponse(
        task_id=task.id,
        filename=task.filename,
        status=task.status,
        stage=task.stage,
        progress=task.progress,
        message=task.message,
        parse_status=result.status if result is not None else None,
        chunk_count=result.chunk_count if result is not None else None,
        elapsed_ms=result.elapsed_ms if result is not None else None,
        errors=result.errors if result is not None else (),
        created_at=task.created_at,
        updated_at=task.updated_at,
    )

from fastapi import FastAPI

from enterprise_rag_engine.api.exception_handlers import (
    enterprise_rag_error_handler,
    unhandled_error_handler,
)
from enterprise_rag_engine.api.repositories.tasks import InMemoryTaskRepository
from enterprise_rag_engine.api.routers.documents import router as documents_router
from enterprise_rag_engine.api.routers.health import router as health_router
from enterprise_rag_engine.api.services.documents import (
    DocumentTaskService,
    LocalDocumentStorage,
    ParserRegistry,
)
from enterprise_rag_engine.exceptions import EnterpriseRagError
from enterprise_rag_engine.logging import configure_logging
from enterprise_rag_engine.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        summary="Production-minded backend for enterprise retrieval-augmented generation.",
    )
    app.add_exception_handler(EnterpriseRagError, enterprise_rag_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    app.state.document_task_service = DocumentTaskService(
        repository=InMemoryTaskRepository(),
        storage=LocalDocumentStorage(settings.upload_dir),
        parser_registry=ParserRegistry(),
        max_upload_bytes=settings.max_upload_bytes,
    )
    app.include_router(health_router)
    app.include_router(documents_router)
    return app


app = create_app()

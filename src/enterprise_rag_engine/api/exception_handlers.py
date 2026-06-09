import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from enterprise_rag_engine.api.schemas.error import ErrorResponse
from enterprise_rag_engine.exceptions import EnterpriseRagError

logger = logging.getLogger(__name__)


async def enterprise_rag_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, EnterpriseRagError):
        return await unhandled_error_handler(request, exc)

    logger.warning(
        "handled application error path=%s error_code=%s message=%s",
        request.url.path,
        exc.error_code,
        exc.message,
    )
    payload = ErrorResponse(error_code=exc.error_code, message=exc.message)
    return JSONResponse(status_code=int(exc.status_code), content=payload.model_dump())


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error path=%s", request.url.path)
    payload = ErrorResponse(error_code="INTERNAL_ERROR", message="Internal server error")
    return JSONResponse(status_code=500, content=payload.model_dump())

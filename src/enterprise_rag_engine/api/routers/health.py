from fastapi import APIRouter

from enterprise_rag_engine.api.schemas.health import HealthResponse
from enterprise_rag_engine.api.services.health import get_health_status

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return get_health_status()

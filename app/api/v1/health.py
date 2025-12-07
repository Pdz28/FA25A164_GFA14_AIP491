"""API v1 health and status routes."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import ServiceDep, SettingsDep
from app.api.schemas import HealthResponse
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: ServiceDep,
    settings: SettingsDep,
) -> HealthResponse:
    """Health check endpoint.
    
    Returns service status and loaded model information.
    
    Args:
        service: Inference service (injected)
        settings: Application settings (injected)
        
    Returns:
        Health status information
    """
    effnet_loaded = getattr(service, "effnet", None) is not None
    swin_loaded = getattr(service, "swin_cls", None) is not None
    
    return HealthResponse(
        ready=True,
        device=str(service.device),
        loaded_checkpoints=getattr(service, "loaded_checkpoints_info", ""),
        effnet_loaded=effnet_loaded,
        swin_loaded=swin_loaded,
        version=settings.app_version,
    )


@router.get("/ping")
async def ping() -> dict:
    """Simple ping endpoint for basic connectivity test."""
    return {"status": "ok"}

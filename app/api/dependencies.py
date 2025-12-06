"""API dependencies for dependency injection."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceNotReadyError
from app.services.inference import InferenceService


def get_service(request: Request) -> InferenceService:
    """Get inference service from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        InferenceService instance
        
    Raises:
        ServiceNotReadyError: If service is not initialized
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise ServiceNotReadyError()
    return service


# Type aliases for dependency injection
ServiceDep = Annotated[InferenceService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

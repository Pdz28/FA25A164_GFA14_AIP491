"""Global exception handlers for FastAPI."""
from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorResponse
from app.core.exceptions import AppException
from app.core.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        """Handle custom application exceptions."""
        logger.warning(
            "app_exception",
            path=request.url.path,
            error=exc.message,
            status_code=exc.status_code,
            details=exc.details,
        )
        
        error_response = ErrorResponse(
            error=exc.message,
            detail=str(exc.details) if exc.details else None,
            status_code=exc.status_code,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(),
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
        )
        
        error_response = ErrorResponse(
            error="Internal server error",
            detail=str(exc) if app.debug else None,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(),
        )

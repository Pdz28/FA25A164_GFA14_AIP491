"""Modern FastAPI application with professional architecture."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.middleware import register_exception_handlers
from app.api.v1 import router as v1_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.inference import InferenceService
from checkpoints.load_weight import load_weight
import logging
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.ERROR)
# Initialize settings and logging
settings = get_settings()
setup_logging(log_level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    logger.info("app_startup", version=settings.app_version)
    
    # Download checkpoints from Hugging Face if needed
    try:
        weight_path = load_weight(checkpoints_dir=str(settings.checkpoints_dir))
        if weight_path:
            logger.info("checkpoints_loaded", path=weight_path)
        else:
            logger.warning("checkpoints_not_loaded", message="Using pretrained backbone checkpoints only")
    except Exception as e:
        logger.warning("checkpoints_download_failed", error=str(e))
    
    # Initialize inference service
    try:
        app.state.service = InferenceService(checkpoints_dir=str(settings.checkpoints_dir))
        logger.info(
            "service_initialized",
            device=str(app.state.service.device),
            checkpoints=app.state.service.loaded_checkpoints_info,
        )
    except Exception as e:
        logger.exception("service_initialization_failed", error=str(e))
        raise
    
    yield
    
    # Cleanup
    logger.info("app_shutdown")
    if hasattr(app.state, "service"):
        service = app.state.service
        if hasattr(service, "gradcam"):
            try:
                service.gradcam.remove_hooks()
                logger.debug("gradcam_hooks_removed")
            except Exception:
                pass


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Professional CNN-Swin Fusion API for skin cancer classification with GradCAM visualization",
        lifespan=lifespan,
        debug=settings.debug,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Exception handlers
    register_exception_handlers(app)
    
    # API routes
    app.include_router(v1_router)
    
    # Static files and templates
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    templates = Jinja2Templates(directory=str(settings.templates_dir))
    app.state.templates = templates
    
    # Legacy routes for backward compatibility
    from app.api.legacy import router as legacy_router
    app.include_router(legacy_router)
    
    # Utility routes
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Avoid 404 noise for favicon requests."""
        return Response(status_code=204)
    
    @app.get("/flutter_service_worker.js", include_in_schema=False)
    async def flutter_sw():
        """Minimal service worker to avoid PWA-related 404s."""
        js = (
            "// noop service worker\n"
            "self.addEventListener('install', e => { self.skipWaiting && self.skipWaiting(); });\n"
            "self.addEventListener('activate', e => { });\n"
            "self.addEventListener('fetch', e => { });\n"
        )
        return Response(content=js, media_type="application/javascript")
    
    @app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
    async def chrome_devtools():
        """Chrome DevTools probe endpoint."""
        return {}
    
    logger.info("app_created", routes=len(app.routes))
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_delay=1.0,
        reload_excludes=[
            "__pycache__",
            ".venv",
            "app/static/outputs",
            "app/static/uploads",
            ".pytest_cache",
        ],
    )
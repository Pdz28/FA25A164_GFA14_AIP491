"""Legacy routes for backward compatibility.

These routes maintain compatibility with the old API structure.
New clients should use /api/v1 endpoints.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.api.dependencies import ServiceDep, SettingsDep
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["legacy"])


@router.get("/")
async def index(request: Request):
    """Serve main UI page."""
    templates = request.app.state.templates
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/health")
async def health_legacy(service: ServiceDep, settings: SettingsDep):
    """Legacy health endpoint (non-standard format)."""
    effnet_loaded = getattr(service, "effnet", None) is not None
    return {
        "ready": True,
        "device": str(service.device),
        "loaded_checkpoints": getattr(service, "loaded_checkpoints_info", ""),
        "effnet_loaded": bool(effnet_loaded),
    }


@router.post("/predict")
async def predict_legacy(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Query("fusion"),
    token_stage: str = Query("7"),
    enhance: bool = Query(False),
    per_pixel: bool = Query(False),
    alpha_min: float = Query(0.0),
    alpha_max: float = Query(0.6),
    service: ServiceDep = None,
    settings: SettingsDep = None,
):
    """Legacy predict endpoint."""
    logger.info("legacy_predict", mode=mode, filename=file.filename)
    
    # Save upload
    filename = file.filename or "upload.png"
    name, _ = os.path.splitext(filename)
    safe_name = name.replace(" ", "_")
    upload_path = settings.upload_dir / f"{safe_name}.png"
    
    img = Image.open(file.file).convert("RGB")
    img.save(upload_path)
    
    try:
        result = service.predict_with_gradcam(
            img,
            str(settings.output_dir),
            mode=mode,
            token_stage=token_stage,
            enhance=enhance,
            per_pixel=per_pixel,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("prediction_error", error=str(e))
        return JSONResponse({"error": "Internal server error", "detail": str(e)}, status_code=500)
    
    def to_url(p: str) -> str:
        rel_path = os.path.relpath(p, settings.static_dir)
        return f"/static/{rel_path.replace(os.sep, '/')}"
    
    return {
        "pred_label": result["pred_label"],
        "probs": result["probs"],
        "uploaded_url": to_url(str(upload_path)),
        "gradcam_url": to_url(result["gradcam_path"]),
    }


@router.post("/predict_all_models")
async def predict_all_models_legacy(
    request: Request,
    file: UploadFile = File(...),
    token_stage: str = Query("7"),
    enhance: bool = Query(False),
    per_pixel: bool = Query(False),
    alpha_min: float = Query(0.0),
    alpha_max: float = Query(0.6),
    service: ServiceDep = None,
    settings: SettingsDep = None,
):
    """Legacy multi-model prediction endpoint."""
    logger.info("legacy_predict_all", filename=file.filename)
    
    filename = file.filename or "upload.png"
    name, _ = os.path.splitext(filename)
    safe_name = name.replace(" ", "_")
    upload_path = settings.upload_dir / f"{safe_name}.png"
    
    img = Image.open(file.file).convert("RGB")
    img.save(upload_path)
    
    def to_url(p: str) -> str:
        rel_path = os.path.relpath(p, settings.static_dir)
        return f"/static/{rel_path.replace(os.sep, '/')}"
    
    results = {}
    models = ["fusion", "effnet", "swin"]
    
    for mode in models:
        try:
            kwargs = {
                "mode": mode,
                "enhance": enhance,
                "per_pixel": per_pixel,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
            }
            if mode == "fusion":
                kwargs["token_stage"] = token_stage
            
            result = service.predict_with_gradcam(img, str(settings.output_dir), **kwargs)
            results[mode] = {
                "pred_label": result["pred_label"],
                "probs": result["probs"],
                "uploaded_url": to_url(str(upload_path)),
                "gradcam_url": to_url(result["gradcam_path"]),
                "error": None,
            }
        except Exception as e:
            logger.warning(f"{mode}_prediction_failed", error=str(e))
            results[mode] = {
                "error": str(e),
                "pred_label": None,
                "probs": None,
                "uploaded_url": to_url(str(upload_path)),
                "gradcam_url": None,
            }
    
    return results

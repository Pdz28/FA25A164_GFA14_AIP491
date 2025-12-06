"""API v1 routes for prediction endpoints."""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from PIL import Image

from app.api.dependencies import ServiceDep, SettingsDep
from app.api.schemas import (
    MultiModelPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.core.exceptions import ModelNotLoadedError, PredictionError
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["prediction"])


def _save_upload(file: UploadFile, settings: SettingsDep) -> tuple[Image.Image, str]:
    """Save uploaded file and return image and URL path.
    
    Args:
        file: Uploaded file
        settings: Application settings
        
    Returns:
        Tuple of (PIL Image, URL path)
    """
    filename = file.filename or "upload.png"
    safe_name = Path(filename).stem.replace(" ", "_")
    unique_id = uuid.uuid4().hex[:8]
    upload_filename = f"{safe_name}_{unique_id}.png"
    upload_path = settings.upload_dir / upload_filename
    
    img = Image.open(file.file).convert("RGB")
    img.save(upload_path)
    
    # Return URL path relative to static
    url_path = f"/static/uploads/{upload_filename}"
    return img, url_path


def _to_url(path: str | Path, settings: SettingsDep) -> str:
    """Convert file path to URL path.
    
    Args:
        path: File path
        settings: Application settings
        
    Returns:
        URL path
    """
    path = Path(path)
    rel_path = path.relative_to(settings.static_dir)
    return f"/static/{rel_path.as_posix()}"


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    request: PredictionRequest = PredictionRequest(),
    service: ServiceDep = None,
    settings: SettingsDep = None,
) -> PredictionResponse:
    """Make prediction with GradCAM visualization.
    
    Args:
        file: Uploaded image file
        request: Prediction parameters
        service: Inference service (injected)
        settings: Application settings (injected)
        
    Returns:
        Prediction result with GradCAM overlay
        
    Raises:
        ModelNotLoadedError: If required model is not loaded
        PredictionError: If prediction fails
    """
    start_time = time.time()
    
    logger.info(
        "prediction_request",
        mode=request.mode,
        filename=file.filename,
        token_stage=request.token_stage
    )
    
    # Save upload
    img, upload_url = _save_upload(file, settings)
    
    try:
        # Run prediction
        result = service.predict_with_gradcam(
            img,
            str(settings.output_dir),
            mode=request.mode,
            token_stage=request.token_stage,
            enhance=request.enhance,
            per_pixel=request.per_pixel,
            alpha_min=request.alpha_min,
            alpha_max=request.alpha_max,
        )
    except RuntimeError as e:
        msg = str(e)
        if "not loaded" in msg.lower() or "not available" in msg.lower():
            logger.warning("model_not_loaded", model=request.mode, error=msg)
            raise ModelNotLoadedError(request.mode)
        logger.error("prediction_failed", error=msg)
        raise PredictionError(msg)
    except Exception as e:
        logger.exception("prediction_error", error=str(e))
        raise PredictionError(str(e))
    
    processing_time = (time.time() - start_time) * 1000
    
    logger.info(
        "prediction_success",
        pred_label=result["pred_label"],
        processing_time_ms=processing_time
    )
    
    return PredictionResponse(
        pred_label=result["pred_label"],
        pred_idx=result["pred_idx"],
        confidence=max(result["probs"].values()),
        probs=result["probs"],
        uploaded_url=upload_url,
        gradcam_url=_to_url(result["gradcam_path"], settings),
        mode=request.mode,
        processing_time_ms=processing_time,
    )


@router.post("/predict/fusion", response_model=PredictionResponse)
async def predict_fusion(
    file: UploadFile = File(...),
    token_stage: str = "7",
    enhance: bool = False,
    per_pixel: bool = False,
    alpha_min: float = 0.0,
    alpha_max: float = 0.6,
    service: ServiceDep = None,
    settings: SettingsDep = None,
) -> PredictionResponse:
    """Fusion model prediction (shortcut endpoint)."""
    request = PredictionRequest(
        mode="fusion",
        token_stage=token_stage,
        enhance=enhance,
        per_pixel=per_pixel,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    return await predict(file, request, service, settings)


@router.post("/predict/effnet", response_model=PredictionResponse)
async def predict_effnet(
    file: UploadFile = File(...),
    enhance: bool = False,
    per_pixel: bool = False,
    alpha_min: float = 0.0,
    alpha_max: float = 0.6,
    service: ServiceDep = None,
    settings: SettingsDep = None,
) -> PredictionResponse:
    """EfficientNet model prediction (shortcut endpoint)."""
    request = PredictionRequest(
        mode="effnet",
        enhance=enhance,
        per_pixel=per_pixel,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    return await predict(file, request, service, settings)


@router.post("/predict/swin", response_model=PredictionResponse)
async def predict_swin(
    file: UploadFile = File(...),
    enhance: bool = False,
    per_pixel: bool = False,
    alpha_min: float = 0.0,
    alpha_max: float = 0.6,
    service: ServiceDep = None,
    settings: SettingsDep = None,
) -> PredictionResponse:
    """Swin Transformer model prediction (shortcut endpoint)."""
    request = PredictionRequest(
        mode="swin",
        enhance=enhance,
        per_pixel=per_pixel,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    return await predict(file, request, service, settings)


@router.post("/predict/all", response_model=MultiModelPredictionResponse)
async def predict_all_models(
    file: UploadFile = File(...),
    token_stage: str = "7",
    enhance: bool = False,
    per_pixel: bool = False,
    alpha_min: float = 0.0,
    alpha_max: float = 0.6,
    service: ServiceDep = None,
    settings: SettingsDep = None,
) -> MultiModelPredictionResponse:
    """Run prediction on all available models.
    
    Args:
        file: Uploaded image file
        token_stage: Token stage for fusion model
        enhance: Enable enhanced visualization
        per_pixel: Enable per-pixel alpha blending
        alpha_min: Minimum alpha value
        alpha_max: Maximum alpha value
        service: Inference service (injected)
        settings: Application settings (injected)
        
    Returns:
        Predictions from all models
    """
    logger.info("multi_model_prediction", filename=file.filename)
    
    # Save upload once
    img, upload_url = _save_upload(file, settings)
    
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
            
            result = service.predict_with_gradcam(
                img,
                str(settings.output_dir),
                **kwargs
            )
            
            results[mode] = PredictionResponse(
                pred_label=result["pred_label"],
                pred_idx=result["pred_idx"],
                confidence=max(result["probs"].values()),
                probs=result["probs"],
                uploaded_url=upload_url,
                gradcam_url=_to_url(result["gradcam_path"], settings),
                mode=mode,
            )
            
        except Exception as e:
            logger.warning(f"{mode}_prediction_failed", error=str(e))
            results[mode] = {
                "error": str(e),
                "pred_label": None,
                "probs": None,
                "uploaded_url": upload_url,
                "gradcam_url": None,
            }
    
    return MultiModelPredictionResponse(
        fusion=results.get("fusion", {}),
        effnet=results.get("effnet", {}),
        swin=results.get("swin", {}),
        uploaded_url=upload_url,
    )

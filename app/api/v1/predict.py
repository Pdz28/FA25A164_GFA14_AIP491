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
from app.core.exceptions import AppException, InvalidInputError, ModelNotLoadedError, PredictionError
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


@router.post("/upload")
async def upload_directory(
    directory_path: str,
    recursive: bool = False,
    extensions: list[str] | None = None,
) -> dict:
    """Scan a directory for images and report count.

    This endpoint does not upload files to the server; it validates a server-side
    directory and counts files matching provided extensions.
    """
    p = Path(directory_path)
    if not p.exists():
        raise AppException("Directory not found", status_code=404, details={"path": directory_path})
    if not p.is_dir():
        raise InvalidInputError("Path is not a directory", details={"path": directory_path})

    exts = set(e.lower() for e in (extensions or [".png", ".jpg", ".jpeg"]))
    pattern_iter = p.rglob("*") if recursive else p.glob("*")
    count = 0
    for fp in pattern_iter:
        if fp.is_file() and fp.suffix.lower() in exts:
            count += 1
    return {"count": count, "message": f"Found {count} files"}


@router.get("/upload")
async def upload_directory_get(
    directory_path: str,
    recursive: bool = False,
    extensions: list[str] | None = None,
) -> dict:
    """GET alias for directory scan to support browser testing."""
    return await upload_directory(directory_path, recursive, extensions)


@router.post("/gradcam")
async def gradcam_generate(
    file: UploadFile | None = File(default=None),
    image_path: str | None = None,
    mode: str = "fusion",
    token_stage: str = "7",
    save_dir: str | None = None,
    return_image_base64: bool = False,
    enhance: bool = False,
    per_pixel: bool = False,
    alpha_min: float = 0.0,
    alpha_max: float = 0.6,
    *,
    service: ServiceDep,
    settings: SettingsDep,
) -> dict:
    """Generate Grad-CAM for a single image.

    Accepts either an uploaded file or a server-side image path.
    """
    if file is None and not image_path:
        raise InvalidInputError("Provide either 'file' or 'image_path'")

    # Prepare image and paths
    if file is not None:
        img, uploaded_url = _save_upload(file, settings)
    else:
        src = Path(image_path)
        if not src.exists():
            raise AppException("Image not found", status_code=404, details={"path": image_path})
        img = Image.open(src).convert("RGB")
        uploaded_url = _to_url(src, settings) if str(src).startswith(str(settings.static_dir)) else str(src)

    out_dir = save_dir or str(settings.output_dir)
    kwargs = {
        "mode": mode,
        "token_stage": token_stage if mode == "fusion" else None,
        "enhance": enhance,
        "per_pixel": per_pixel,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
    }
    # Remove None to avoid unexpected params
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    result = service.predict_with_gradcam(img, out_dir, **kwargs)

    payload = {
        "pred_idx": result["pred_idx"],
        "pred_label": result["pred_label"],
        "probs": result["probs"],
        "orig_path": uploaded_url,
        "gradcam_path": _to_url(result["gradcam_path"], settings),
    }

    if return_image_base64:
        import base64
        with open(result["gradcam_path"], "rb") as f:
            payload["overlay_base64"] = base64.b64encode(f.read()).decode("utf-8")

    return payload


@router.get("/gradcam")
async def gradcam_generate_get(
    image_path: str,
    mode: str = "fusion",
    token_stage: str = "7",
    save_dir: str | None = None,
    return_image_base64: bool = False,
    enhance: bool = False,
    per_pixel: bool = False,
    alpha_min: float = 0.0,
    alpha_max: float = 0.6,
    *,
    service: ServiceDep,
    settings: SettingsDep,
) -> dict:
    """GET alias for Grad-CAM when using a server-side `image_path`."""
    return await gradcam_generate(
        file=None,
        image_path=image_path,
        mode=mode,
        token_stage=token_stage,
        save_dir=save_dir,
        return_image_base64=return_image_base64,
        enhance=enhance,
        per_pixel=per_pixel,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        service=service,
        settings=settings,
    )


@router.post("/load_weights")
async def load_weights(
    weights_dir: str,
    *,
    service: ServiceDep,
) -> dict:
    """Load or reload model weights from a directory.

    The directory may contain files like best_fusion_model.pth, best_effnetb0.pth, best_swin.pth
    or LoRA adapters. This reuses the service loader to refresh models.
    """
    p = Path(weights_dir)
    if not p.exists() or not p.is_dir():
        raise AppException("Invalid path", status_code=400, details={"path": weights_dir})

    try:
        service._load_model(str(p))  # refresh in-place
    except Exception as e:
        raise AppException("Checkpoint not found or failed to load", status_code=404, details={"error": str(e)})

    return {
        "message": "Weights loaded",
        "loaded_weights_info": getattr(service, "loaded_checkpoints_info", ""),
        "device": str(service.device),
        "classes": getattr(service, "class_names", []),
    }


@router.get("/load_weights")
async def load_weights_get(
    weights_dir: str | None = None,
    *,
    service: ServiceDep,
    settings: SettingsDep,
) -> dict:
    """GET alias for loading weights; defaults to `settings.checkpoints_dir` when omitted."""
    target = weights_dir or str(settings.checkpoints_dir)
    return await load_weights(target, service)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    request: PredictionRequest = PredictionRequest(),
    *,
    service: ServiceDep,
    settings: SettingsDep,
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
    *,
    service: ServiceDep,
    settings: SettingsDep,
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
    *,
    service: ServiceDep,
    settings: SettingsDep,
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
    *,
    service: ServiceDep,
    settings: SettingsDep,
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
    *,
    service: ServiceDep,
    settings: SettingsDep,
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

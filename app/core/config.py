"""Application configuration management."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "CNN-Swin Fusion API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    weights_dir: Path = base_dir / "weights"
    static_dir: Path = base_dir / "app" / "static"
    templates_dir: Path = base_dir / "app" / "templates"
    upload_dir: Path = static_dir / "uploads"
    output_dir: Path = static_dir / "outputs"
    
    # Model
    device: Literal["cuda", "cpu", "auto"] = "auto"
    num_classes: int = 2
    img_size: tuple[int, int] = (224, 224)
    class_names: list[str] = ["benign", "malignant"]
    
    # Hugging Face
    hf_repo_id: str = "PDZ2810/b3_swin_fusion_skin-cancer"
    hf_weight_file: str = "best_hybrid_model.pth"
    hf_token: str | None = None
    
    # GradCAM
    default_gradcam_mode: str = "fusion"
    default_token_stage: str = "7"
    default_alpha: float = 0.45
    default_alpha_min: float = 0.0
    default_alpha_max: float = 0.6
    
    # CORS
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

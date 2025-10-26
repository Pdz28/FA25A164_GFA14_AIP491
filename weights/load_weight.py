from __future__ import annotations

import os
from typing import Optional


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    return v.strip()


def load_weight(
    *,
    weights_dir: str,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    token_env: str = "HF_TOKEN",
    repo_env: str = "HF_REPO_ID",
    file_env: str = "HF_WEIGHT_FILE",
) -> Optional[str]:
    """
    Ensure a .pth file exists in `weights_dir` by downloading it from Hugging Face Hub.

    Priority for parameters:
    - Explicit function args `repo_id`, `filename` (if provided)
    - Environment variables HF_REPO_ID, HF_WEIGHT_FILE
    - Project defaults (repo uploaded by user)

    Returns the local file path if present (existing or downloaded) else None.
    """
    # Defaults based on user's repo
    repo_id = repo_id or _get_env(repo_env, "PDZ2810/b3_swin_fusion_skin-cancer")
    filename = filename or _get_env(file_env, "best_cnnswin_lora_binary_continue.pth")

    if not repo_id or not filename:
        return None

    os.makedirs(weights_dir, exist_ok=True)
    dest_path = os.path.join(weights_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    # Lazy import to avoid hard dependency at import time
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required. Please add it to requirements.txt and install."
        ) from e

    token = _get_env(token_env)
    try:
        # Download into the weights directory (no symlinks to shared cache)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=weights_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        return local_path
    except Exception as e:
        # Non-fatal: log and continue startup without weights
        print(f"[hfhub] Download failed for {repo_id}/{filename}: {e}")
        return None

# Fusion Model FastAPI Prototype (EfficientNet-B3 + Swin-T + LoRA)

A minimal FastAPI app that serves a CNN+Swin fusion model and visualizes GradCAM overlays.

Features
- Upload an image and get a prediction (binary: benign / malignant).
- Visualize GradCAM overlays produced from the CNN branch and (optionally) an EffNetB0 visualizer.
- Swin token saliency (PatchCAM) visualization is supported; the token stage is now fixed to the final token stage (77) to match the simplified UI.
- Perpixel alpha and percentile normalization (enhance mode) are available to improve overlay contrast.
- A lightweight `/health` endpoint reports service readiness and whether EffNet visuals are available.

Project structure (important files)
```
main.py                 # FastAPI entrypoint and routes (use this one)
app/
    models/
        cnnswin.py       # CNNViTFusion model
        cnn_b0.py        # EffNet-B0 helper (optional visualization)
    services/
        inference.py     # Preprocessing, inference, Grad-CAM generation
    utils/
        gradcam.py       # Grad-CAM utilities and overlay helpers
    templates/
        index.html       # Web UI
    static/
        css/, js/, uploads/, outputs/
weights/                # Put checkpoints here (see notes)
requirements.txt
README.md
```

Quick setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Weights and EffNet visuals
- The server looks for checkpoint files in the `weights/` folder. If you have a combined fusion checkpoint, name it something like `best_cnnswin_lora_binary_continue.pth` (or any `.pth`) and place it in `weights/`.
- EffNetB0 visualizations are optional. To enable them, place an EffNet checkpoint whose filename starts with `eff` (for example `best_effnetb0.pth`) into the `weights/` folder. On startup the app will attempt to load any matching `eff*.pth` file and enable the `effnet` visualization mode in the UI.
- If no checkpoints are present the app will still start and use backbone defaults (ImageNet weights)  predictions will be nontrained but the UI and GradCAM overlay demos still function.

Run the server (local development)
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# or simply
python main.py
```
Open http://127.0.0.1:8000/ and use the UI to upload images and visualize overlays.

API highlights
- POST /predict  Form upload with file and query params:
  - mode: `fusion` (default), `cnn`, `swin_patchcam`, `effnet`
  - token_stage: fixed to `7` in the UI and backend defaults to last stage (77)
  - enhance: boolean (percentile normalization + per-pixel alpha when helpful)
  - per_pixel: boolean (use per-pixel alpha blending)
  - alpha_min / alpha_max: per-pixel alpha bounds when `per_pixel=true`
- GET /health  Returns JSON describing service readiness, loaded weights name, device, and whether EffNet visualization is available.

Frontend and Vercel deployment notes
- This repository contains two frontends:
  - A simple Jinja2-based UI served by FastAPI (default when running `main.py`).
  - A Next.js frontend inside the `frontend/` folder intended for Vercel deployments (App Router). If you deploy to Vercel, set the project Root Directory to `frontend/` so Vercel builds the Next.js app instead of the Python server.
- For the Next.js Edge proxy to the Hugging Face Inference API, set the required environment variables in Vercel (or your hosting provider):
  - `HF_API_TOKEN`  your Hugging Face token (if using private repo/models)
  - `HF_INFERENCE_URL` or `HF_REPO_ID` depending on how you host weights; check `frontend/app/api/predict/route.js` for the exact expectation.

Notes and recent UI decisions
- The posthoc crossattention (`fusion_attn`) visualization branch has been removed (deprecated) to simplify the UI.
- The Swin token stage selector was removed and the token stage is fixed to the last stage (77). This avoids confusion and keeps overlays consistent with the fusion visuals.
- Use `enhance=true` to apply percentile clipping (default 298%) before normalizing the saliency map; this often improves visibility for lowcontrast heatmaps.
- The UI now displays a small status line (queried from `/health`) indicating whether weights were loaded and whether the EffNet visualizer is available.

Troubleshooting
- If the UI says "EffNet: unavailable": check `weights/` for a matching `eff*.pth` file and restart the server.
- If the server prints processor/transform loader errors (401 when fetching HF artifacts), make sure any HF token in the environment is correct or unset it  the server falls back to ImageNet mean/std when necessary.
- PyTorch CPU-only runs can be slow for GradCAM backward passes. For local testing with reasonable speed, use a CUDA-capable environment.

If you'd like, I can also add a short README section describing commands to prepare a tiny test image and run a quick local inference to verify both `fusion` and `effnet` modes.

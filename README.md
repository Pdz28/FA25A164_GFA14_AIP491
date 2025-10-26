# Fusion Model FastAPI Prototype (EfficientNet-B3 + Swin-T + LoRA)

This is a minimal FastAPI app that serves a trained CNN+Swin fusion model and visualizes Grad-CAM with an interactive slider.

It provides a simple web UI with two panes:
- Left: the uploaded image
- Right: a before/after slider comparing the original vs. the Grad-CAM overlay (similar to the slider shown in the provided screenshot)

## Project structure

```
main.py                 # FastAPI entrypoint and routes (use this one)
app/
	models/
		cnnswin.py            # CNNViTFusion model + AttentionFusion (inference-ready)
	services/
		inference.py          # Preprocessing, inference, Grad-CAM generation, file I/O
	utils/
		gradcam.py            # Lightweight Grad-CAM implementation (CNN branch)
	templates/
		index.html            # Web UI with two panes and slider
	static/
		css/styles.css        # Styling (dark theme)
		js/app.js             # Frontend logic (upload, fetch, slider)
		uploads/              # Saved uploads
		outputs/              # Generated Grad-CAM overlays
weights/
	best_cnnswin_lora_binary_continue.pth  # (optional) Put your trained weights here
requirements.txt
README.md
```

## Setup (Windows PowerShell)

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place your trained checkpoint as `weights/best_cnnswin_lora_binary_continue.pth` (or any .pth; the app scans the folder). If not present, the app will still start using ImageNet pre-trained backbones (predictions will be non-sensical, but the Grad-CAM UI works).

## Run the server

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 and:
1) Upload an image (benign/malignant skin lesion, etc.)
2) Click Analyze
3) Use the slider to reveal the Grad-CAM overlay

## Notes

- There is also a duplicate src/app/main.py from a previous "src" layout. Ignore it and use the root `main.py` entrypoint above to avoid path issues for static/templates.

- Grad-CAM is computed on the last convolutional block of the EfficientNet-B3 branch, then overlaid on the original image.
- The Swin branch is used for prediction and fusion, but CAM visualization focuses on the CNN spatial feature maps for clarity.
- If your adapter weights were saved separately during training (e.g., `adapter_best_epochXX/`), you can ignore them if you have the combined `best_cnnswin_lora_binary.pth`. The state dict includes LoRA layers.

## Troubleshooting

- CUDA is optional. If a GPU is available, it will be used automatically; otherwise CPU will be used.
- If you see dependency or build errors on torch/torchvision, install the versions compatible with your Python and CUDA setup. See PyTorch.org for the exact wheel.


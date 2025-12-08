# CNN-Swin Fusion Model - Complete System

Professional skin cancer classification system with hybrid CNN+Transformer architecture, featuring FastAPI inference server, GradCAM visualization, and streamlined training pipeline.

**Highlights**
- FastAPI inference API with `/api/v1` routes and web UI
- GradCAM visualization for Fusion, CNN, Swin PatchCAM, EfficientNet
- Swin-Tiny training script with your augmentation pipeline
- Checkpoints-first boot; clean fallback to pretrained weights
- Windows-friendly setup and training

**Project Structure**
```
├── main.py                      # FastAPI server entry point
├── pyproject.toml               # Project metadata (optional)
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
├── SYSTEM_UPDATES.md            # Changelog / notes
├── app/                         # Backend application (FastAPI)
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/                  # API v1 endpoints
│   │       ├── __init__.py
│   │       └── (routers).py     # Health, predict routes
│   ├── core/
│   │   ├── __init__.py
│   │   └── (config).py          # Settings, logging, exceptions
│   ├── models/                  # Model definitions
│   │   ├── __init__.py
│   │   ├── cnn_b0.py            # EfficientNet-B0 backbone
│   │   ├── swin.py              # Swin Tiny model + dataset
│   │   └── cnnswin.py           # Fusion model (CNN+Swin)
│   ├── services/
│   │   └── inference.py         # Inference modes + GradCAM
│   ├── utils/
│   │   └── gradcam.py           # Grad-CAM utilities
│   ├── templates/
│   │   └── index.html           # Web UI template
│   └── static/                  # Frontend assets and output dirs
│       ├── css/
│       │   └── styles.css
│       ├── js/
│       │   └── app.js
│       ├── uploads/             # User-uploaded images (ignored)
│       └── outputs/             # Visualization outputs (ignored)
├── training/                    # Training scripts and data
│   ├── train_hybrid_model.py                 # Fusion training (all-in-one)
│   ├── train_swin_tiny.py       # Swin-Tiny training with augmentation
│   ├── train_efficientnetb0.py  # EfficientNet-B0 training script
│   └── data/                    # Datasets (ignored)
│       ├── train/
│       │   ├── benign/
│       │   └── malignant/
│       └── valid/
│           ├── benign/
│           └── malignant/
    ├── checkpoints/                 # Model weights for runtime
    │   ├── best_swin.pth
    │   ├── best_effnetb0.pth
    │   └── hybrid_model.pth         # Optional fusion checkpoint
├── weights/                     # Legacy or external weights
│   ├── __init__.py
│   ├── best_effnetb0.pth
│   ├── best_hybrid_model.pth
│   ├── best_swin.pth
│   └── load_weight.py
├── frontend/                    # Optional Next.js UI
│   ├── package.json
│   ├── next.config.js
│   └── app/
│       ├── layout.js
│       ├── page.js
│       └── api/predict/route.js
├── scripts/                     # Utility scripts (optional)
│   └── README.md
├── src/                         # Library-style code (optional)
│   ├── __init__.py
│   └── core/
│       └── __init__.py
└── static/                      # Public static assets (optional)
  └── js/
    └── app.js
```

**Folder Details**
- `app/api/v1`: Health check and predict endpoints; extend here for new APIs.
- `app/models`: All PyTorch modules: CNN, Swin, Fusion; replace or add backbones here.
- `app/services/inference.py`: Loads checkpoints/adapters; implements `fusion|cnn|swin_patchcam|effnet` modes.
- `app/utils/gradcam.py`: Grad-CAM helpers for CNN and EfficientNet.
- `training`: Standalone training scripts; outputs can be stored under `training/save_checkpoints/`.
- `checkpoints`: Inference-time weights loaded by the server.
- `weights`: Legacy directory; safe to keep for archival but not required.
- `frontend`: Optional Next.js client. Server works without it.
- `static`: Extra frontend assets; not required for API-only usage.

**Setup (Windows PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Run Server**
```powershell
python main.py                # Dev (auto-reload via settings)
uvicorn main:app --reload     # Direct uvicorn
python main.py --prod         # Production (no reload)
```

**API Routes**
- `GET /api/v1/health`: Device, checkpoints info, readiness
- `POST /api/v1/predict`: Image upload + visualization
  - `mode`: `fusion|cnn|swin_patchcam|effnet`
  - `enhance`, `per_pixel`, `alpha_min`, `alpha_max`

**Visualization Modes**
- `fusion`: CNN+Swin fused attention
- `cnn`: CNN Grad-CAM (EfficientNet visualizer)
- `swin_patchcam`: Swin token saliency (final stage)
- `effnet`: EfficientNet Grad-CAM (optional)

**Checkpoints**
- Place weights in `checkpoints/`
  - Fusion: `hybrid_model.pth` (optional)
  - Swin: `best_swin.pth`
  - EffNet: `best_effnetb0.pth`
- Loader uses `strict=False`; falls back to pretrained if missing

**Training - Swin Tiny**
- File: `training/train_swin_tiny.py`
- Augmentation (applied before processor):
  - Resize(224), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.2)
  - RandomRotation(30, bilinear), ColorJitter(0.2/0.2/0.15/0.05)
  - RandomAffine(translate=0.1, scale 0.9–1.1, shear=5)
- Processor: `AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")`
- Saves: `checkpoints/best_swin.pth`
- Run:
```powershell
python training\train_swin_tiny.py
```

**Training - EfficientNet-B0**
- File: `training/train_efficientnetb0.py`
- Class-weighted loss + `WeightedRandomSampler`
- Fix tqdm outside notebooks: `from tqdm.auto import tqdm`
- Saves to `training/save_checkpoints/efficientnetb0/`

**Tips & Troubleshooting**
- If Swin processor fails (HF issues), code falls back to ImageNet stats
- CPU-only: reduce `batch_size`; use `num_workers=0`, `pin_memory=False`
- Ensure dataset layout:
  - `training/data/train/{benign,malignant}/...`
  - `training/data/valid/{benign,malignant}/...`
- To stop tracking uploads/outputs: see `.gitignore` and run `git rm -r --cached ...`
 - Training checkpoints: prefer storing intermediate runs in `training/save_checkpoints/` to keep `checkpoints/` clean for production.

**License & Credits**
- MIT License
- EfficientNet (Google), Swin (Microsoft), LoRA (Microsoft), FastAPI, PyTorch

Built with ❤️ for skin cancer classification research.

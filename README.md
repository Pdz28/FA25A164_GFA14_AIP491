# CNN-Swin Fusion Model - Complete System

Professional skin cancer classification system with hybrid CNN+Transformer architecture, featuring FastAPI inference server, GradCAM visualization, and streamlined training pipeline.

## 🎯 Features

### Inference & Visualization
- **FastAPI server** with real-time prediction endpoint
- **GradCAM visualization** with multiple modes: fusion, CNN, Swin PatchCAM, EfficientNet
- **Interactive web UI** for image upload and overlay visualization
- **Multi-mode support**: fusion (default), cnn, swin_patchcam, effnet
- **Enhanced visualization**: Percentile normalization and per-pixel alpha blending
- **Health monitoring**: `/health` endpoint for service status

### Model Architecture
- **EfficientNet-B0**: CNN backbone for low-level feature extraction
- **Swin-Tiny**: Transformer backbone for global context
- **AttentionFusion**: Advanced fusion with spatial tokens, positional embeddings, and dynamic gating
- **LoRA adapters**: Parameter-efficient fine-tuning for Swin
- **Multi-head prediction**: Fusion head + auxiliary CNN/Swin heads

### Training
- **Simple training script**: All-in-one file (~300 LOC)
- **Progressive unfreezing**: Backbone unfreezing at epoch 8
- **Automatic checkpointing**: Save best model based on validation loss
- **Early stopping**: Patience-based stopping (40 epochs)
- **Data augmentation**: Flip, Rotation, ColorJitter

## 📁 Project Structure

```
├── main.py                    # FastAPI server entry point
├── app/
│   ├── main.py               # Application factory
│   ├── api/
│   │   ├── v1/              # API v1 endpoints
│   │   ├── legacy.py        # Backward compatibility
│   │   └── schemas.py       # Pydantic models
│   ├── core/
│   │   ├── config.py        # Settings management
│   │   ├── logging.py       # Structured logging
│   │   └── exceptions.py    # Custom exceptions
│   ├── models/
│   │   ├── cnnswin.py       # CNNViTFusion (main model)
│   │   └── cnn_b0.py        # EfficientNet-B0 visualizer
│   ├── services/
│   │   └── inference.py     # Inference service with GradCAM
│   ├── utils/
│   │   └── gradcam.py       # GradCAM utilities
│   ├── templates/
│   │   └── index.html       # Web UI
│   └── static/
│       ├── css/, js/        # Frontend assets
│       └── uploads/, outputs/
├── training/
│   ├── train.py             # Simple training script (all-in-one)
│   └── README.md            # Training documentation
├── checkpoints/
│   ├── best_hybrid_model.pth    # Main fusion checkpoint
│   ├── best_effnetb0.pth        # Optional EfficientNet visualizer
│   └── load_weight.py           # Weight loading utilities
├── frontend/                # Next.js frontend (Vercel deployment)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Setup

### Installation (Windows PowerShell)

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### For Linux/Mac

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 💪 Model checkpoints

The system uses checkpoints from the `checkpoints/` folder:

### Main Fusion Model
- **Required**: Place your trained fusion model as `best_hybrid_model.pth` in `checkpoints/`
- **Auto-loading**: System prioritizes: `best_hybrid_model.pth` → `best_swin.pth` → `best_effnetb0.pth`
- **Fallback**: If no checkpoint exists, uses ImageNet pretrained checkpoints (predictions will be untrained but system functions)

### Optional EfficientNet Visualizer
- **Enable GradCAM**: Place EfficientNet checkpoint starting with `eff*` (e.g., `best_effnetb0.pth`) in `checkpoints/`
- **UI Integration**: When available, "effnet" mode appears in visualization dropdown
- **Check status**: Visit `/health` endpoint to see if EfficientNet visualizer is loaded

### Weight File Priority
```python
# Priority order for fusion model:
1. best_hybrid_model.pth      # Recommended name
2. best_swin.pth               # Fallback 1
3. best_effnetb0.pth           # Fallback 2
```

## 🏃 Running the Server

### Development Mode (with auto-reload)

```bash
# Simple way
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# With multiple workers
python main.py --prod --workers 4

# Custom host/port
python main.py --host 0.0.0.0 --port 8080

# Set log level
python main.py --log-level info
```

### CLI Options

```
--host HOST       Server host (default: 0.0.0.0)
--port PORT       Server port (default: 8000)
--prod           Production mode (no reload)
--workers N      Number of workers (default: 1)
--log-level LVL  Log level (debug, info, warning, error)
```

### Access the Application

- **Web UI**: http://127.0.0.1:8000/
- **API Docs**: http://127.0.0.1:8000/docs (Swagger UI)
- **ReDoc**: http://127.0.0.1:8000/redoc
- **Health Check**: http://127.0.0.1:8000/api/v1/health

## 📡 API Reference

### POST /api/v1/predict

Main prediction endpoint with GradCAM visualization.

**Form Parameters:**
- `file`: Image file (JPG, PNG)
- `mode`: Visualization mode (default: `fusion`)
  - `fusion`: Combined CNN+Swin attention
  - `cnn`: CNN branch GradCAM
  - `swin_patchcam`: Swin token saliency
  - `effnet`: EfficientNet visualizer (if available)
- `enhance`: Enable percentile normalization (boolean, default: false)
- `per_pixel`: Use per-pixel alpha blending (boolean, default: false)
- `alpha_min`: Minimum alpha for per-pixel mode (float, 0.0-1.0)
- `alpha_max`: Maximum alpha for per-pixel mode (float, 0.0-1.0)

**Response:**
```json
{
  "prediction": "benign",
  "confidence": 0.87,
  "probabilities": {
    "benign": 0.87,
    "malignant": 0.13
  },
  "overlay_path": "/outputs/overlay_xxx.jpg",
  "original_path": "/uploads/xxx.jpg"
}
```

### GET /api/v1/health

Service health and status check.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "checkpoints_file": "best_hybrid_model.pth",
  "device": "cuda",
  "effnet_available": true
}
```

### Legacy Endpoints (Backward Compatibility)

- `POST /predict` → redirects to `/api/v1/predict`
- `GET /health` → redirects to `/api/v1/health`

## 🎓 Training

### Simple Training Script

The training module is streamlined into a single file for easy understanding and customization.

```bash
# Run training with default config
python training/train.py
```

### Configuration

Edit `training/train.py` to customize training parameters:

```python
class Config:
    # Data paths
    train_dir = "/path/to/train"
    val_dir = "/path/to/val"
    
    # Hyperparameters
    batch_size = 32
    epochs = 101
    
    # Learning rates
    classifier_lr = 2e-4
    lora_lr = 5e-5
    backbone_lr_unfrozen = 1e-4
    
    # LoRA config
    lora_r = 16
    lora_alpha = 32
    
    # Training schedule
    effnet_unfreeze_epoch = 8
    early_stopping_patience = 40
```

### Training Features

- ✅ **Progressive Unfreezing**: EfficientNet backbone unfreezes at epoch 8
- ✅ **LoRA Fine-tuning**: Efficient Swin adaptation (~300K trainable params)
- ✅ **Automatic Checkpointing**: Saves best model to `checkpoints/best_model.pth`
- ✅ **Early Stopping**: Stops if no improvement for 40 epochs
- ✅ **Data Augmentation**: Flip, Rotation, ColorJitter
- ✅ **Parameter Groups**: Different LR for backbone/classifier/LoRA

### Training Output

```
Epoch 10/101
  Train: Loss=0.3245 | Acc=85.67%
  Val:   Loss=0.3891 | Acc=82.34%
  🎉 Best model saved!
```

### Using Trained checkpoints

After training completes, copy the checkpoint for inference:

```bash
# Copy best model to checkpoints folder
cp checkpoints/best_model.pth checkpoints/best_hybrid_model.pth

# Restart the server to load new checkpoints
python main.py
```

## 🌐 Deployment

### Frontend Options

This repository contains two frontend implementations:

#### 1. FastAPI + Jinja2 (Default)
- **Built-in**: Served directly by FastAPI
- **Location**: `app/templates/index.html`
- **Use case**: Simple deployment, development
- **Access**: http://localhost:8000/

#### 2. Next.js (Vercel Deployment)
- **Location**: `frontend/` folder
- **Framework**: Next.js 14 with App Router
- **Use case**: Production deployment on Vercel
- **Setup**: Set Vercel Root Directory to `frontend/`

### Vercel Deployment

1. **Connect Repository**: Link your GitHub repo to Vercel
2. **Set Root Directory**: `frontend/`
3. **Environment Variables**:
   ```
   HF_API_TOKEN=your_huggingface_token
   HF_INFERENCE_URL=your_inference_endpoint
   ```
4. **Deploy**: Vercel auto-builds the Next.js app

### Environment Variables

```bash
# .env file
HF_API_TOKEN=hf_xxxxx              # Hugging Face token (optional)
HF_INFERENCE_URL=https://...       # Inference endpoint (optional)
```

## 🎨 UI Features & Updates

### Current UI Simplifications

- ✅ **Fixed token stage**: Swin PatchCAM uses final stage (77) for consistency
- ✅ **Removed deprecated modes**: `fusion_attn` visualization removed
- ✅ **Enhanced visualization**: Percentile normalization improves low-contrast heatmaps
- ✅ **Status indicator**: Shows loaded checkpoints and EfficientNet visualizer availability
- ✅ **Per-pixel alpha**: Advanced blending option for fine-tuned overlays

### Visualization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `fusion` | Combined CNN+Swin attention (default) | Best overall visualization |
| `cnn` | CNN branch GradCAM only | Focus on low-level features |
| `swin_patchcam` | Swin token saliency | Transformer attention patterns |
| `effnet` | EfficientNet visualizer | Alternative CNN perspective |

### Enhancement Options

- **Enhance Mode**: Applies percentile clipping (2-98%) before normalization
- **Per-pixel Alpha**: Uses gradient magnitude for variable transparency
- **Alpha Range**: Custom min/max bounds for per-pixel blending

## 🔧 Troubleshooting

### Common Issues

#### "EffNet: unavailable" in UI
- **Cause**: No EfficientNet visualizer checkpoint found
- **Solution**: Place `eff*.pth` file in `checkpoints/` folder and restart server

#### Model not loading / Using ImageNet checkpoints
- **Cause**: No checkpoint files in `checkpoints/` folder
- **Solution**: 
  1. Train a model: `python training/train.py`
  2. Copy checkpoint: `cp checkpoints/best_model.pth checkpoints/best_hybrid_model.pth`
  3. Restart server

#### Slow inference on CPU
- **Cause**: GradCAM backward passes are CPU-intensive
- **Solution**: Use CUDA-enabled environment for reasonable speed
- **Alternative**: Disable GradCAM for faster inference (set `mode=None`)

#### HuggingFace 401 errors
- **Cause**: Invalid or missing HF token when loading transformers
- **Solution**: 
  - Unset `HF_API_TOKEN` environment variable
  - Server will fall back to ImageNet normalization

#### Out of Memory during training
- **Cause**: Batch size too large for available RAM/VRAM
- **Solution**: Reduce `batch_size` in `training/train.py` Config class

#### ZeroDivisionError during training
- **Cause**: Empty dataset or incorrect data paths
- **Solution**: Verify `train_dir` and `val_dir` paths in Config

### Debug Mode

Run server with debug logging:

```bash
python main.py --log-level debug
```

### Health Check

Verify system status:

```bash
curl http://localhost:8000/api/v1/health
```

## 🧪 Testing

### Quick Test

1. **Prepare test image**: Place any skin lesion image in a temp folder
2. **Test inference**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/predict \
     -F "file=@test_image.jpg" \
     -F "mode=fusion"
   ```
3. **Check output**: Look for overlay image in `app/static/outputs/`

### Test All Modes

```bash
# Test fusion mode
curl -F "file=@test.jpg" -F "mode=fusion" http://localhost:8000/api/v1/predict

# Test CNN mode
curl -F "file=@test.jpg" -F "mode=cnn" http://localhost:8000/api/v1/predict

# Test Swin PatchCAM
curl -F "file=@test.jpg" -F "mode=swin_patchcam" http://localhost:8000/api/v1/predict

# Test EfficientNet (if available)
curl -F "file=@test.jpg" -F "mode=effnet" http://localhost:8000/api/v1/predict
```

## 📊 Model Architecture Details

### CNNViTFusion

```
Input (224x224x3)
     ↓
├─→ EfficientNet-B0 (CNN Branch)
│      ├─ 5.3M params
│      └─ Feature maps: 1280 channels
│
├─→ Swin-Tiny (Transformer Branch)
│      ├─ 28M params (base)
│      ├─ LoRA: ~300K trainable params
│      └─ Tokens: 49 patches (7x7)
│
└─→ AttentionFusion
       ├─ Spatial token attention
       ├─ Positional embeddings
       ├─ Dynamic gating (alpha)
       ├─ Multi-head cross-attention
       └─ Output: fused features
            ↓
       Classifier Heads
       ├─ Main: Fusion → 2 classes
       ├─ Aux CNN: CNN features → 2 classes
       └─ Aux Swin: Swin features → 2 classes
```

### Parameter Breakdown

| Component | Parameters | Trainable (Initial) | Trainable (After Epoch 8) |
|-----------|------------|---------------------|---------------------------|
| EfficientNet-B0 | 5.3M | ❄️ Frozen | ✅ Active |
| Swin-Tiny (base) | 28M | ❄️ Frozen | ❄️ Frozen |
| LoRA adapters | 300K | ✅ Active | ✅ Active |
| Fusion module | 7M | ✅ Active | ✅ Active |
| Classifier heads | 2M | ✅ Active | ✅ Active |
| **TOTAL** | **42.6M** | **9.3M (22%)** | **14.6M (34%)** |

## 📚 Documentation

- **Training**: See `training/README.md`
- **API Architecture**: See `ARCHITECTURE_SUMMARY.md`
- **Professional Version**: See `README_V2.md` (detailed API docs)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

MIT License

## 🙏 Acknowledgments

- **EfficientNet**: Google Research
- **Swin Transformer**: Microsoft Research
- **LoRA**: Microsoft Research
- **FastAPI**: Sebastián Ramírez
- **PyTorch**: Meta AI

---

**Built with ❤️ for skin cancer classification research**

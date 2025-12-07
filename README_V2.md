# Professional CNN-Swin Fusion API

A production-ready FastAPI application for skin cancer classification using hybrid CNN-Swin Transformer fusion with GradCAM visualization.

## 🏗️ Architecture

### Modern Professional Structure
```
app/
├── main.py                 # Application factory & entry point
├── core/
│   ├── config.py          # Centralized configuration management
│   ├── logging.py         # Structured logging setup
│   └── exceptions.py      # Custom exception classes
├── api/
│   ├── v1/
│   │   ├── predict.py     # Prediction endpoints
│   │   └── health.py      # Health check endpoints
│   ├── dependencies.py    # Dependency injection
│   ├── schemas.py         # Pydantic request/response models
│   ├── middleware.py      # Exception handlers
│   └── legacy.py          # Backward-compatible routes
├── models/                 # ML model implementations
│   ├── cnnswin.py         # Fusion model (new architecture)
│   ├── cnn_b0.py          # EfficientNet-B0
│   └── swin.py            # Swin Transformer
├── services/
│   └── inference.py       # Inference service layer
└── utils/
    └── gradcam.py         # GradCAM utilities

checkpoints/                    # Model checkpoints
main.py                     # Legacy entry point (backward compatible)
main_v2.py                  # New entry point
.env.example               # Environment configuration template
```

## ✨ Key Improvements

### 1. **Configuration Management**
- Centralized settings via `pydantic-settings`
- Environment variable support with `.env` file
- Type-safe configuration with validation

### 2. **Structured Logging**
- JSON-formatted logs via `structlog`
- Contextual logging with metadata
- Request tracing and error tracking

### 3. **Professional API Design**
- **Versioned endpoints** (`/api/v1/...`)
- **Pydantic schemas** for request/response validation
- **Dependency injection** for cleaner code
- **Custom exception handling** with proper HTTP status codes

### 4. **Error Handling**
- Custom exception hierarchy
- Global exception handlers
- Detailed error responses with timestamps
- Proper logging of all errors

### 5. **Type Safety**
- Full type hints throughout
- Pydantic models for data validation
- Better IDE support and autocomplete

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment
```powershell
# Copy example config
cp .env.example .env

# Edit .env with your settings
# Set HF_TOKEN if using private Hugging Face repos
```

### 3. Run Server

**Option A: New Entry Point (Recommended)**
```powershell
python main_v2.py
# or
uvicorn app.main:app --reload
```

**Option B: Legacy Entry Point (Backward Compatible)**
```powershell
python main.py
```

## 📡 API Endpoints

### V1 API (New)

#### Health Check
```http
GET /api/v1/health
GET /api/v1/ping
```

#### Single Model Prediction
```http
POST /api/v1/predict
POST /api/v1/predict/fusion
POST /api/v1/predict/effnet
POST /api/v1/predict/swin
```

**Request Body:**
```json
{
  "mode": "fusion",
  "token_stage": "7",
  "enhance": false,
  "per_pixel": false,
  "alpha_min": 0.0,
  "alpha_max": 0.6
}
```

**Response:**
```json
{
  "pred_label": "malignant",
  "pred_idx": 1,
  "confidence": 0.87,
  "probs": {
    "benign": 0.13,
    "malignant": 0.87
  },
  "uploaded_url": "/static/uploads/image.png",
  "gradcam_url": "/static/outputs/gradcam_xxx.png",
  "mode": "fusion",
  "processing_time_ms": 245.3
}
```

#### Multi-Model Prediction
```http
POST /api/v1/predict/all
```

Returns predictions from all three models (fusion, effnet, swin).

### Legacy API (Backward Compatible)

```http
GET  /health
POST /predict
POST /predict_all_models
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | CNN-Swin Fusion API | Application name |
| `DEBUG` | false | Debug mode |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `DEVICE` | auto | Device (cuda/cpu/auto) |
| `HF_REPO_ID` | PDZ2810/... | Hugging Face repo |
| `HF_WEIGHT_FILE` | best_hybrid_model.pth | Weight filename |
| `HF_TOKEN` | - | HF API token |
| `LOG_LEVEL` | INFO | Logging level |
| `LOG_FORMAT` | json | Log format (json/text) |

## 🎯 Model Architecture

### Fusion Model (Primary)
- **CNN Branch**: EfficientNet-B0 (spatial features)
- **Transformer Branch**: Swin-Tiny (semantic features)
- **Fusion**: Advanced AttentionFusion with:
  - Spatial token-based attention
  - Learnable positional embeddings
  - Dynamic gating mechanism
  - Gradient-weighted fusion
  - LoRA fine-tuning on Swin

### Standalone Models
- **EfficientNet-B0**: CNN-only baseline
- **Swin-Tiny**: Transformer-only baseline

## 📊 GradCAM Visualization Modes

1. **fusion**: Gradient-weighted combination of CNN CAM + Swin token saliency
2. **cnn**: CNN Grad-CAM only
3. **effnet**: Standalone EfficientNet visualization
4. **swin**: Swin token saliency
5. **swin_patchcam**: Multi-resolution token saliency

## 🧪 Testing

```powershell
# Health check
curl http://localhost:8000/api/v1/health

# Prediction
curl -X POST http://localhost:8000/api/v1/predict/fusion \
  -F "file=@image.jpg" \
  -F "enhance=true"
```

## 📝 Development

### Code Quality
- Type hints throughout
- Docstrings for all public functions
- Structured logging
- Exception handling

### Best Practices
- Dependency injection
- Separation of concerns
- Configuration management
- Error handling hierarchy

## 🔄 Migration from Old API

Old clients using `/predict` and `/health` will continue to work via the legacy router. 

**Recommended migration path:**
1. Update clients to use `/api/v1/...` endpoints
2. Use new response models with additional fields
3. Enable structured logging
4. Configure via `.env` file

## 📚 Documentation

- **Interactive API Docs**: http://localhost:8000/docs (when DEBUG=true)
- **ReDoc**: http://localhost:8000/redoc (when DEBUG=true)

## 🔐 Production Deployment

```powershell
# Set environment
$env:DEBUG="false"
$env:LOG_FORMAT="json"
$env:HOST="0.0.0.0"
$env:PORT="8000"

# Run with production server
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

## 📦 Dependencies

- **FastAPI**: Modern web framework
- **Pydantic**: Data validation
- **structlog**: Structured logging
- **PyTorch**: Deep learning
- **Transformers**: Swin model
- **PEFT**: LoRA fine-tuning

---

**Version**: 1.0.0  
**License**: MIT  
**Author**: Your Team

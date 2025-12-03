# CNN-ViT Fusion System Updates

## Changes Made

### 1. Updated Model Architecture (`app/models/cnnswin.py`)
- Added `CrossModelAttention` class for attention-based fusion
- Updated `CNNViTFusion` to use attention fusion instead of simple concatenation  
- Swin features query CNN features through multi-head cross-attention
- Maintains `forward_with_tokens` for Grad-CAM visualization compatibility

### 2. Created Training Notebook (`fusion_training_notebook.ipynb`)
- Complete training pipeline with progressive unfreezing strategy
- Cross-model attention fusion with Swin querying CNN features
- Dual data augmentation pipelines for EfficientNet and Swin branches
- Advanced training techniques: MixUp, label smoothing, gradient clipping
- WandB integration for experiment tracking
- Progressive unfreezing: CNN frozen during warmup, then last 5 blocks unfrozen

### 3. Training Strategy
- **Warmup phase** (3 epochs): Only Swin + Attention + Classifier training
- **Main phase**: Add CNN (last 5 blocks) with separate learning rates:
  - CNN backbone: 1e-4
  - Swin transformer: 1e-4  
  - Classifier/Attention: 2e-4
  - LayerNorm/Bias: 5e-5
- Linear warmup → Cosine annealing scheduler
- MixUp augmentation applied to both image branches simultaneously

### 4. Inference Compatibility
- Existing inference service works with new attention fusion
- Grad-CAM still generates fusion visualization using gradient weighting
- `forward_with_tokens` maintains same interface for token visualization

## Usage

### Training
1. Update data paths in notebook configuration section
2. Uncomment training loop in notebook
3. Run training cells
4. Monitor with WandB dashboard

### Inference  
No changes needed - existing inference service automatically works with attention fusion model.

### Key Benefits
- **Better feature interaction**: Swin can dynamically attend to CNN features
- **Improved fusion**: Learned attention weights vs simple concatenation  
- **Maintained visualization**: Grad-CAM fusion still works
- **Progressive training**: Stable learning with gradual complexity increase
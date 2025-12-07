"""
Simple Training Script for CNN-Swin Fusion
===========================================
Simplified training with essential features only.

Usage:
    python training/train.py
"""
import os
import sys
import time
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Ensure project root (repo root) is importable when running from training/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from app.models.cnnswin import CNNViTFusion
from peft import get_peft_model, LoraConfig


class Config:
    # Data/Model
    num_classes = 2
    img_size = (224, 224)
    train_dir = os.path.join('training/data/train')
    val_dir = os.path.join('training/data/val')
    batch_size = 16
    num_workers = 0

    # Training
    epochs = 100
    seed = 42

    # Optimizer
    backbone_lr = 5e-5
    backbone_lr_unfrozen = 1e-4
    classifier_lr = 2e-4
    lora_lr = 5e-5
    weight_decay = 1e-4

    # LoRA
    lora_r = 16
    lora_alpha = 32

    # Scheduling
    label_smoothing = 0.08
    effnet_unfreeze_epoch = 1
    backbone_warmup_epochs = 8
    swin_stage3_unfreeze_epoch = 2

    # Checkpointing
    checkpoint_dir = "training/checkpoint"
    early_stopping_patience = 40


config = Config()


# ============================================================================
# Dataset
# ============================================================================
class DualBranchDataset(torch.utils.data.Dataset):
    """Dataset that returns two versions of the same image."""
    
    def __init__(self, root, transform_common=None, transform_branch=None):
        self.dataset = datasets.ImageFolder(root)
        self.transform_common = transform_common
        self.transform_branch = transform_branch
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if self.transform_common:
            img = self.transform_common(img)
        
        if self.transform_branch:
            img_cnn = self.transform_branch(img)
            img_swin = self.transform_branch(img)
        else:
            img_cnn = transforms.ToTensor()(img)
            img_swin = transforms.ToTensor()(img)
        
        return img_cnn, img_swin, label


def get_dataloaders():
    """Create train and validation dataloaders."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Train transforms
    train_common = transforms.Compose([
        transforms.Resize(config.img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.15, 0.05),
    ])
    
    train_branch = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Val transforms
    val_common = transforms.Resize(config.img_size, interpolation=InterpolationMode.BILINEAR)
    val_branch = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Datasets
    train_dataset = DualBranchDataset(config.train_dir, train_common, train_branch)
    val_dataset = DualBranchDataset(config.val_dir, val_common, val_branch)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False  # Set False to avoid warning when no GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False  # Set False to avoid warning when no GPU
    )
    
    return train_loader, val_loader


# ============================================================================
# Training Functions
# ============================================================================
def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_lora(model):
    """Apply LoRA to Swin attention and freeze non-LoRA params."""
    try:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["query", "key", "q_proj", "k_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model.swin = get_peft_model(model.swin, lora_config)
        # Only LoRA params trainable at start
        for name, param in model.swin.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("✅ LoRA applied")
    except Exception as e:
        print(f"⚠️ LoRA failed: {e}")
    return model


def print_detailed_stats(model):
    print("\n" + "="*70)
    print("📊 Parameter Information")
    print("="*70)

    # Khởi tạo tổng số tham số trainable
    total_trainable = 0

    # --- 1. TÍNH FUSION MODULE (Luôn Trainable) ---
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    fusion_trainable = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
    total_trainable += fusion_trainable
    print(f"🧱 Fusion Module (Heads/Gate/Attn): {fusion_params:>15,} | Trainable: {fusion_trainable:>15,}")

    # --- 2. TÍNH CNN BACKBONE (Giả sử Đã Unfreeze Toàn bộ) ---
    cnn_backbone_params = sum(p.numel() for p in model.cnn_backbone.parameters())
    cnn_backbone_trainable = sum(p.numel() for p in model.cnn_backbone.parameters() if p.requires_grad)
    total_trainable += cnn_backbone_trainable
    print(f"🐘 CNN Backbone (EffNet):           {cnn_backbone_params:>15,} | Trainable: {cnn_backbone_trainable:>15,}")
    
    # --- 3. TÍNH SWIN VÀ LORA ---
    swin_total = 0
    lora_total = 0
    lora_trainable = 0
    swin_base_trainable = 0  # Swin Base đang được huấn luyện (không gồm LoRA)

    for n, p in model.swin.named_parameters():
        cnt = p.numel()
        swin_total += cnt
        nl = n.lower()
        if ('lora' in nl) or ('adapter' in nl):
            lora_total += cnt
            if p.requires_grad:
                lora_trainable += cnt
                total_trainable += cnt
        elif p.requires_grad:
            swin_base_trainable += cnt
            total_trainable += cnt

    # LORA (Luôn được tính vào trainable nếu p.requires_grad=True)
    

    # 4. TÍNH TỪNG STAGE CỦA SWIN (Tập trung vào Base/Trainable)
    try:
        swin_layers = resolve_swin_layers(model.swin)
        
        # Hàm tính tham số BASE/Trainable (không tính LORA) trong một layer
        def get_base_trainable(layer):
            count = 0
            for n, p in layer.named_parameters():
                if ("lora" not in n.lower() and "adapter" not in n.lower()) and p.requires_grad:
                    count += p.numel()
            return count

        stage1_base_trainable = get_base_trainable(swin_layers[0]) if len(swin_layers) > 0 else 0
        stage2_base_trainable = get_base_trainable(swin_layers[1]) if len(swin_layers) > 1 else 0
        stage3_base_trainable = get_base_trainable(swin_layers[2]) if len(swin_layers) > 2 else 0
        stage4_base_trainable = get_base_trainable(swin_layers[3]) if len(swin_layers) > 3 else 0

        print("\n--- 🧠 SWIN BASE (Without LoRA) ---")
        # Stage 1 & 2 thường bị đóng băng (Stage 3 mở ở E18+)
        print(f"  ├─ Stage 1 Base (E0):           {stage1_base_trainable:>15,} (Status: FROZEN)")
        print(f"  ├─ Stage 2 Base (E0):           {stage2_base_trainable:>15,} (Status: FROZEN)")
        print(f"  ├─ Stage 3 Base (E18+):         {stage3_base_trainable:>15,} (Status: UNFREEZE)")
        print(f"  └─ Stage 4 Base (E125+):        {stage4_base_trainable:>15,} (Status: FROZEN)") 
        
    except Exception as e:
        print(f"  ⚠️ Không đếm được Stage do cấu trúc lạ: {e}")
    print(f" LoRA Param (Added):               {lora_total:>15,} | Trainable: {lora_trainable:>15,}")

    # --- 5. TỔNG KẾT ---
    total_all = sum(p.numel() for p in model.parameters())
    
    print("-" * 70)
    print(f"📌 Toatal Params:   {total_all:>15,}")
    print(f"🔓 Total TRAINABLE Params (Final Count):   {total_trainable:>15,} ({100*total_trainable/total_all:.2f}%)")
    print("="*70 + "\n")


def resolve_swin_layers(swin_module):
    """Return the Swin encoder layers ModuleList regardless of PEFT wrapping."""
    paths = [
        'base_model.model.swin.encoder.layers',
        'base_model.model.encoder.layers',
        'model.swin.encoder.layers',
        'model.encoder.layers',
        'encoder.layers',
    ]
    for path in paths:
        obj = swin_module
        ok = True
        for attr in path.split('.'):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError('Could not locate Swin encoder layers path')


def create_optimizer(model):
    """Create optimizer with parameter groups."""
    params = {
        "lora": [],
        "classifier": [],
        "backbone": [],
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'lora' in name.lower():
            params["lora"].append(param)
        elif any(k in name.lower() for k in ['classifier', 'head', 'fusion', 'gate']):
            params["classifier"].append(param)
        elif 'cnn_backbone' in name.lower() or 'swin' in name.lower():
            params["backbone"].append(param)
    
    param_groups = []
    if params["lora"]:
        param_groups.append({"params": params["lora"], "lr": config.lora_lr, "weight_decay": 0.0})
    if params["classifier"]:
        param_groups.append({"params": params["classifier"], "lr": config.classifier_lr, "weight_decay": config.weight_decay})
    if params["backbone"]:
        param_groups.append({"params": params["backbone"], "lr": config.backbone_lr, "weight_decay": config.weight_decay})
    
    return torch.optim.AdamW(param_groups)


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.epochs}")
    for x_cnn, x_swin, targets in pbar:
        x_cnn, x_swin, targets = x_cnn.to(device), x_swin.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, alpha, cnn_logits, swin_logits = model(x_cnn, x_swin)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * targets.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    if total == 0:
        print("⚠️ Warning: No samples processed in training epoch")
        return 0.0, 0.0
    
    return total_loss / total, 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_cnn, x_swin, targets in tqdm(loader, desc="Validation"):
            x_cnn, x_swin, targets = x_cnn.to(device), x_swin.to(device), targets.to(device)
            
            logits, _, _, _ = model(x_cnn, x_swin)
            loss = criterion(logits, targets)
            
            total_loss += loss.item() * targets.size(0)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    if total == 0:
        print("⚠️ Warning: No samples processed in validation")
        return 0.0, 0.0
    
    return total_loss / total, 100. * correct / total


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("CNN-Swin Fusion Training")
    print("="*60)
    
    # Setup
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders()
    print(f"✅ Train: {len(train_loader.dataset)} samples")
    print(f"✅ Val: {len(val_loader.dataset)} samples\n")
    
    # Model
    print("Creating model...")
    model = CNNViTFusion(num_classes=config.num_classes, img_size=config.img_size)
    model = setup_lora(model)
    
    # Freeze backbones initially
    for param in model.cnn_backbone.parameters():
        param.requires_grad = False
    for name, param in model.swin.named_parameters():
        if 'lora' in name.lower():
            # keep LoRA trainable
            continue
        param.requires_grad = False
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}\n")
    
    # Training
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = create_optimizer(model)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...\n")
    for epoch in range(1, config.epochs + 1):
        # Unfreeze EfficientNet backbone at scheduled epoch
        if epoch == config.effnet_unfreeze_epoch:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}: Unfreezing EfficientNet backbone")
            print(f"{'='*60}")
            for param in model.cnn_backbone.parameters():
                param.requires_grad = True
            optimizer = create_optimizer(model)
            
            # Print trainable params breakdown
            print_detailed_stats(model)
            print(f"{'='*60}\n")
        
        # Unfreeze Swin Stage 3 at scheduled epoch
        if epoch == config.swin_stage3_unfreeze_epoch:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}: Unfreezing Swin Transformer Stage 3")
            print(f"{'='*60}")
            # Unfreeze stage 3 (layers[2])
            try:
                layers = resolve_swin_layers(model.swin)
                # Enable train on stage 3 parameters
                for name, param in layers[2].named_parameters():
                    param.requires_grad = True
                print("✅ Swin Stage 3 unfrozen")
            except Exception as e:
                print(f"⚠️ Could not unfreeze Swin Stage 3: {e}")
            
            optimizer = create_optimizer(model)
            
            # Print trainable params breakdown
            print_detailed_stats(model)
            print(f"{'='*60}\n")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print
        print(f"\nEpoch {epoch}/{config.epochs}")
        print(f"  Train: Loss={train_loss:.4f} | Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f} | Acc={val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, f"{config.checkpoint_dir}/best_model.pth")
            print(f"  Best model saved!")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        print("-" * 60)
    
    print(f"\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor
from sklearn.metrics import precision_score, recall_score, f1_score

# Ensure project root (repo root) is importable when running from training/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
# Import từ hệ thống (đảm bảo file ở: app/models/swin.py)
from app.models.swintiny import SwinDataset, SwinTinyFull  # nếu bạn vẫn muốn dùng SwinDataset thay ImageFolder, đổi lại dễ thôi

# Config
SEED = 42
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 2
batch_size = 32
epochs = 30
lr = 2e-4
weight_decay = 1e-4
label_smoothing = 0.05
img_size = 224

# Dữ liệu (đổi đường dẫn nếu cần)
train_dir = "training/data/train"
val_dir   = "training/data/val"
ckpt_dir  = "training/save_checkpoints/swintiny"
ckpt_path = os.path.join(ckpt_dir, "swintiny.pth")
os.makedirs(ckpt_dir, exist_ok=True)

# Processor (khởi tạo nhưng transforms manual đang xử lý preprocessing)
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# ImageNet stats
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
img_size = 224
# Transforms

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30, interpolation=InterpolationMode.BILINEAR, expand=False),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

def build_dataloaders():
    # Kiểm tra lớp
    tmp = datasets.ImageFolder(train_dir)
    classes = tmp.classes
    print(f"[DATA] Classes: {classes}")

    # Dùng ImageFolder với transform
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds   = datasets.ImageFolder(val_dir, transform=val_transform)

    # Windows/CPU an toàn: num_workers=0, pin_memory=False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader

def metrics(y_true_t, logits_t):
    y_true = y_true_t.cpu().numpy()
    y_pred = logits_t.argmax(dim=1).cpu().numpy()
    acc = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="binary", zero_division=0)
    return acc, prec, rec, f1

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits)
            all_labels.append(labels)
    val_loss = total_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc, prec, rec, f1 = metrics(all_labels, all_logits)
    return {"val_loss": val_loss, "val_acc": acc, "val_prec": prec, "val_rec": rec, "val_f1": f1}

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())
    train_loss = total_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc, prec, rec, f1 = metrics(all_labels, all_logits)
    return {"train_loss": train_loss, "train_acc": acc, "train_prec": prec, "train_rec": rec, "train_f1": f1}

def main():
    train_loader, val_loader = build_dataloaders()
    model = SwinTinyFull(num_classes=num_classes, dropout_p=0.4).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=5, min_lr=1e-6)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer)
        va = validate(model, val_loader, criterion)
        scheduler.step(va["val_loss"])

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch:03d}] "
              f"TrainLoss={tr['train_loss']:.4f} Acc={tr['train_acc']:.4f} F1={tr['train_f1']:.4f} | "
              f"ValLoss={va['val_loss']:.4f} Acc={va['val_acc']:.4f} F1={va['val_f1']:.4f} | LR={cur_lr:.2e}")

        if va["val_loss"] < best_val:
            best_val = va["val_loss"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved best Swin checkpoint -> {ckpt_path}")

    print("✅ Done.")

if __name__ == "__main__":
    main()

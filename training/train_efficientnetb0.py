import os
import random
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms, datasets
from torchvision.transforms import InterpolationMode
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------- CONFIG ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = (224, 224)
batch_size = 32
epochs = 100
classifier_lr = 2e-4
weight_decay = 1e-3
num_workers = 0  # nếu chạy Windows, set = 0 nếu gặp lỗi
num_classes = 2

train_input  = "training/data/train"
val_input    = "training/data/val"

ckpt_path = "training/save_checkpoints/efficientnetb0/efficientnetb0.pth"
os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

# ---------------- TRANSFORMS & DATASET ----------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
img_size = 224

transform_effnet_train = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30, interpolation=InterpolationMode.BILINEAR, expand=False),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

transform_effnet_val = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

class SingleDataset(Dataset):
    def __init__(self, folder, transform_eff=None):
        self.ds = datasets.ImageFolder(folder, transform=None)
        self.samples = self.ds.samples
        self.classes = self.ds.classes
        self.transform_eff = transform_eff

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform_eff:
            img_eff = self.transform_eff(img)
        else:
            img_eff = transforms.ToTensor()(img)
        return img_eff, label

# ---------------- DATA & SAMPLER ----------------
train_dataset = SingleDataset(train_input, transform_effnet_train)
val_dataset   = SingleDataset(val_input,   transform_effnet_val)

# Compute class weights (for loss) and per-sample weights (for sampler)
targets = [label for _, label in train_dataset]
class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
samples_weight = np.array([class_weights[t] for t in targets])
samples_weight = torch.from_numpy(samples_weight).double()

sampler = WeightedRandomSampler(
    weights=samples_weight,
    num_samples=len(samples_weight),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

# ---------------- MODEL ----------------
class EffNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = effnet.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1280, num_classes)
 
    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.pool(feats).view(feats.size(0), -1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

model = EffNetClassifier(num_classes=num_classes).to(device)

# Optionally unfreeze backbone (already unfrozen below)
for p in model.backbone.parameters():
    p.requires_grad = True

def print_param_counts(model, name="model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} - Total params: {total:,} | Trainable params: {trainable:,} | Trainable%: {100.*trainable/total:.2f}%")

print_param_counts(model, "EffNetClassifier")

# ---------------- OPTIM & LOSS ----------------
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.3)
optimizer = optim.AdamW(model.parameters(), lr=classifier_lr, weight_decay=weight_decay)
# We'll step scheduler on validation accuracy (mode="max")
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=2, threshold=0.4, min_lr=9e-4)

# ---------------- TRAIN & VALID FUNCTIONS ----------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    loop = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.*correct/total
    epoch_prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_rec  = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_f1   = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    loop = tqdm(loader, desc="Valid", leave=False)
    with torch.no_grad():
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.*correct/total
    epoch_prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_rec  = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_f1   = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1

# ---------------- TRAINING LOOP (train + val) ----------------
best_val_acc = 0.0
history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
}

print("\n🚀 Bắt đầu training (có validation mỗi epoch)...\n")
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}")

    tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)

    print(f"Train | loss {tr_loss:.4f} acc {tr_acc:.2f}% prec {tr_prec:.3f} rec {tr_rec:.3f} f1 {tr_f1:.3f}")
    print(f"Valid | loss {val_loss:.4f} acc {val_acc:.2f}% prec {val_prec:.3f} rec {val_rec:.3f} f1 {val_f1:.3f}")

    # update history
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Scheduler step on validation accuracy
    scheduler.step(val_acc)

    # Save best by validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved best model (val_acc improved -> {best_val_acc:.2f}%)")

print("\n✅ Training finished.")
print("Final best validation accuracy: {:.2f}%".format(best_val_acc))
print("Saved checkpoint:", ckpt_path)

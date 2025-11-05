
import os, random, shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets
from transformers import SwinForImageClassification, AutoImageProcessor
from sklearn.utils.class_weight import compute_class_weight

class SwinDataset(Dataset):
    def __init__(self, folder, processor_swin=None, augment=False):
        self.ds = datasets.ImageFolder(folder, transform=None)
        self.samples = self.ds.samples
        self.classes = self.ds.classes
        self.processor_swin = processor_swin
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.augment:
            img = train_transform(img)

        if self.processor_swin:
            pixel_values = self.processor_swin(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            return pixel_values, label
        else:
            return transforms.ToTensor()(img), label

class SwinTinyFull(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.4):
        super().__init__()
        self.swin = SwinForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # ✅ Thêm dropout trước classifier
        in_features = self.swin.classifier.in_features
        self.swin.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x_swin):
        outputs = self.swin(pixel_values=x_swin)
        return outputs.logits

# Default train-time augmentation used when SwinDataset(augment=True)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
])

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
    def __init__(self, folder, processor_swin=None):
        self.ds = datasets.ImageFolder(folder, transform=None)
        self.samples = self.ds.samples
        self.classes = self.ds.classes
        self.processor_swin = processor_swin

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        pixel_values = self.processor_swin(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label

class SwinTinyFull(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.4):
        super().__init__()
        self.swin = SwinForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        in_features = self.swin.classifier.in_features
        self.swin.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x_swin):
        outputs = self.swin(pixel_values=x_swin)
        return outputs.logits


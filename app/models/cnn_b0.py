import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms, datasets
from transformers import SwinModel, AutoImageProcessor
from sklearn.utils.class_weight import compute_class_weight
from tqdm.notebook import tqdm
import os, random, shutil
import numpy as np
from PIL import Image


class EffNetClassifier(nn.Module):
    def __init__(self, num_classes=2, img_size=(224,224)):
        super().__init__()
        # 🟢 Dùng EfficientNet-B0 và trọng số ImageNet
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = effnet.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1280, num_classes)  # ⚠️ B0 có output dim = 1280 (khác B3 = 1536)

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.pool(feats).view(feats.size(0), -1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

    def get_last_cnn_layer(self) -> nn.Module:
        """
        Return a reference to the last convolutional block of the backbone.

        This is useful for attaching Grad-CAM hooks. Implementation returns the
        last child module of `self.backbone` which in torchvision EfficientNet
        corresponds to the final feature block.
        """
        children = list(self.backbone.children())
        if not children:
            return self.backbone
        return children[-1]

    def compute_gradcam_map(self, input_tensor: torch.Tensor, target_class: int | None = None):
        """
        Lightweight Grad-CAM for EfficientNet-B0.

        Args:
            input_tensor: Tensor shaped (1,3,H,W), preprocessed for the model.
            target_class: optional int class index. If None, uses argmax.

        Returns:
            cam: numpy array float32 in [0,1] sized (H_feature, W_feature) before upsampling.
        """
        # register hooks on last conv
        target_layer = self.get_last_cnn_layer()

        activations = {}
        gradients = {}

        def forward_hook(m, inp, out):
            activations['val'] = out.detach()

        def backward_hook(m, gin, gout):
            gradients['val'] = gout[0].detach()

        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        # ensure gradients are tracked on target layer
        orig_flags = [p.requires_grad for p in target_layer.parameters(recurse=True)]
        for p in target_layer.parameters(recurse=True):
            p.requires_grad_(True)

        try:
            self.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = self.forward(input_tensor)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())
            score = logits[:, target_class].sum()
            score.backward()

            acts = activations.get('val', None)
            grads = gradients.get('val', None)
            if acts is None or grads is None:
                raise RuntimeError('Grad-CAM hooks did not capture activations/gradients')

            # acts: [B,C,H,W], grads: [B,C,H,W]
            weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
            cam = (weights * acts).sum(dim=1, keepdim=False)  # [B,H,W]
            cam = torch.relu(cam)
            cam = cam[0].cpu().float()
            # normalize to 0..1
            mn, mx = cam.min(), cam.max()
            if (mx - mn) < 1e-6:
                cam_np = torch.zeros_like(cam).numpy()
            else:
                cam_np = ((cam - mn) / (mx - mn)).numpy()
            return cam_np
        finally:
            # cleanup hooks and restore flags
            fh.remove()
            bh.remove()
            for p, f in zip(target_layer.parameters(recurse=True), orig_flags):
                p.requires_grad_(f)
# Dataset: chỉ trả ảnh cho EfficientNet
# ----------------------------
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
from __future__ import annotations

import os
import uuid
from typing import Dict, Tuple
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor

from app.models.cnnswin import CNNViTFusion
from app.utils.gradcam import GradCAM


class InferenceService:
    def __init__(self, weights_dir: str, device: str | torch.device = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = ["benign", "malignant"]
        self.img_size = (224, 224)
        self.loaded_weights_info = ""
        self._build_transforms()
        self._load_model(weights_dir)

    def _build_transforms(self):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        # Some environments set an invalid HF token and cause 401 when fetching the processor.
        # Try anonymous first; if it fails, fall back to ImageNet stats.
        try:
            # Force anonymous access and opt into fast processor to avoid warnings
            self.processor = AutoImageProcessor.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224",
                token=False,
                use_fast=True,
            )
        except Exception as e:
            print(f"[swin] Processor load failed, using ImageNet stats fallback: {e}")
            self.processor = SimpleNamespace(image_mean=imagenet_mean, image_std=imagenet_std)

        self.transform_effnet = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        self.transform_swin = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
            ]
        )

    def _load_model(self, weights_dir: str):
        self.model = CNNViTFusion(num_classes=len(self.class_names), img_size=self.img_size)
        self.model.to(self.device).eval()

        # 1) Try full state dict (.pth)
        pth_candidates = []
        default_pth = os.path.join(weights_dir, "best_cnnswin_lora_binary.pth")
        if os.path.exists(default_pth):
            pth_candidates.append(default_pth)
        # any .pth in weights_dir
        if os.path.isdir(weights_dir):
            for fn in os.listdir(weights_dir):
                if fn.lower().endswith(".pth"):
                    path = os.path.join(weights_dir, fn)
                    if path not in pth_candidates:
                        pth_candidates.append(path)

        loaded = False
        for path in pth_candidates:
            try:
                sd = torch.load(path, map_location=self.device)
                self.model.load_state_dict(sd, strict=False)
                self.loaded_weights_info = f"Loaded checkpoint: {os.path.basename(path)}"
                print(self.loaded_weights_info)
                loaded = True
                break
            except Exception as e:
                print(f"Failed to load {path}: {e}")

        # 2) If no full .pth, try LoRA adapter directory
        if not loaded and os.path.isdir(weights_dir):
            adapter_dirs = []
            for name in os.listdir(weights_dir):
                ap = os.path.join(weights_dir, name)
                if os.path.isdir(ap) and (
                    os.path.exists(os.path.join(ap, "adapter_config.json"))
                    or os.path.exists(os.path.join(ap, "adapter_model.bin"))
                ):
                    adapter_dirs.append(ap)

            for ap in adapter_dirs:
                try:
                    # self.model.swin is a PEFT-wrapped model; load adapter weights
                    self.model.swin.load_adapter(ap, adapter_name="default")
                    self.model.swin.set_adapter("default")
                    self.loaded_weights_info = f"Loaded LoRA adapter: {os.path.basename(ap)}"
                    print(self.loaded_weights_info)
                    loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load adapter {ap}: {e}")

        if not loaded:
            self.loaded_weights_info = "No trained weights found; using backbone defaults."
            print(self.loaded_weights_info)

        # Prepare Grad-CAM on the last CNN block
        target_layer = self.model.get_last_cnn_layer()
        self.gradcam = GradCAM(self.model, target_layer)

    @torch.no_grad()
    def preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        x_eff = self.transform_effnet(image).unsqueeze(0).to(self.device)
        x_swin = self.transform_swin(image).unsqueeze(0).to(self.device)
        return x_eff, x_swin

    def _normalize_np(self, x: np.ndarray) -> np.ndarray:
        m, M = float(x.min()), float(x.max())
        if M - m < 1e-8:
            return np.zeros_like(x)
        return (x - m) / (M - m)

    def _percentile_normalize(self, x: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
        """Robust normalize using percentile clipping to enhance contrast.

        Clamps values between the low/high percentiles, then scales to [0,1].
        """
        if not np.isfinite(x).any():
            return np.zeros_like(x)
        lo, hi = np.percentile(x, [low, high])
        if hi - lo < 1e-8:
            return np.zeros_like(x)
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo)

    def predict_with_gradcam(
        self,
        image: Image.Image,
        save_dir: str,
        mode: str = "fusion",
        token_stage: str | None = None,
        enhance: bool = False,
    ) -> Dict:
        os.makedirs(save_dir, exist_ok=True)
        x_eff, x_swin = self.preprocess(image)

        if mode == "cnn":
            # Old behavior: CAM from the CNN branch only (still post-fusion w.r.t. logit)
            logits = self.model(x_eff, x_swin)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            cam = self.gradcam.generate(x_eff, x_swin, target_class=pred_idx, upsample_size=image.size[::-1])
            cam_np = cam.numpy()
            overlay = self.gradcam.overlay_on_image(image, cam_np, alpha=0.45)
        elif mode == "fusion_attn":
            # Post-hoc cross-attention between Swin tokens (Q) and CNN spatial tokens (K,V)
            with torch.no_grad():
                feats = self.model.cnn_backbone(x_eff)
                feats = self.model.cnn_bn(feats)  # [B,C,H,W]
                B, C, H, W = feats.shape
                cnn_tokens = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,Nc,C]

                swin_out = self.model.swin(pixel_values=x_swin)
                swin_tokens = swin_out.last_hidden_state  # [B,Nv,Ds]

                # Project to common dim using the same fusion projections and norms
                q = self.model.fuse.norm_q(self.model.fuse.swin_proj(swin_tokens))  # [B,Nv,D]
                kv = self.model.fuse.norm_kv(self.model.fuse.cnn_proj(cnn_tokens))  # [B,Nc,D]

                # Use the same MHA module to get attention weights
                self.model.eval()
                attn_out, attn_w = self.model.fuse.attn(
                    q, kv, kv, need_weights=True, average_attn_weights=False
                )  # attn_w: [B, heads, Nv, Nc]

                # Average over heads and queries (Nv) to get spatial map over Nc tokens
                attn_map = attn_w.mean(dim=(1, 2))[0]  # [Nc]
                attn_map = attn_map.reshape(H, W).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                attn_map = attn_map / (attn_map.max() + 1e-8)
                attn_up = F.interpolate(attn_map, size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
                cam_np = attn_up.detach().cpu().numpy()

                # For completeness, also compute logits to report prediction
                logits = self.model(x_eff, x_swin)
                probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))

            overlay = self.gradcam.overlay_on_image(image, cam_np, alpha=0.45)
        elif mode == "swin_patchcam":
            # ViT token-only saliency (Patch-CAM at last stage tokens)
            self.model.zero_grad(set_to_none=True)
            # Ensure Swin path builds a grad graph even if model params are frozen
            x_swin.requires_grad_(True)
            with torch.enable_grad():
                # Use selected or higher‑resolution Swin tokens if available
                ts = (token_stage or "hr")
                logits, swin_tokens = self.model.forward_with_tokens(x_eff, x_swin, token_stage=ts)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))

            swin_tokens.retain_grad()
            score = logits[:, pred_idx].sum()
            score.backward()

            token_grads = swin_tokens.grad  # [B,N,D]
            if token_grads is not None:
                # Grad × Activation saliency
                ga = (token_grads[0] * swin_tokens.detach()[0]).sum(dim=-1)  # [N]
                ga = F.relu(ga)
                N = ga.shape[0]
                S = int(N ** 0.5)
                if S * S == N:
                    sal_map = ga.reshape(S, S).unsqueeze(0).unsqueeze(0)
                    sal_up = F.interpolate(sal_map, size=image.size[::-1], mode="bicubic", align_corners=False)[0, 0]
                    cam_np = sal_up.detach().cpu().numpy()
                    cam_np = self._percentile_normalize(cam_np, 2, 98) if enhance else self._normalize_np(cam_np)
                else:
                    cam_np = ga.detach().cpu().numpy()
                    cam_np = self._percentile_normalize(cam_np, 2, 98) if enhance else self._normalize_np(cam_np)
            else:
                cam_np = np.zeros((image.size[1], image.size[0]), dtype=np.float32)

            overlay = self.gradcam.overlay_on_image(
                image,
                cam_np,
                alpha=0.45,
                **({"per_pixel": True, "alpha_min": 0.0, "alpha_max": 0.6} if enhance else {})
            )
        else:
            # Fusion mode: single forward/backward to get both CNN CAM and Swin token saliency
            # Ensure target layer can receive grads
            target_layer = self.model.get_last_cnn_layer()
            orig_flags = [p.requires_grad for p in target_layer.parameters(recurse=True)]
            for p in target_layer.parameters(recurse=True):
                p.requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            # Ensure Swin path builds a grad graph even if model params are frozen
            x_swin.requires_grad_(True)
            with torch.enable_grad():
                # Use selected or higher‑resolution Swin tokens if available
                ts = (token_stage or "hr")
                logits, swin_tokens = self.model.forward_with_tokens(x_eff, x_swin, token_stage=ts)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))

            # retain grad for tokens
            swin_tokens.retain_grad()
            score = logits[:, pred_idx].sum()
            score.backward()

            # Build CNN Grad-CAM from cached hooks (same backward)
            cam_t = self.gradcam.build_from_cached(upsample_size=image.size[::-1])
            cam_np = cam_t.numpy()

            # Swin token saliency -> [N] -> [S,S] -> upsample to image
            token_grads = swin_tokens.grad  # [B,N,D]
            if token_grads is not None:
                # Grad × Activation saliency
                ga = (token_grads[0] * swin_tokens.detach()[0]).sum(dim=-1)  # [N]
                ga = F.relu(ga)
                N = ga.shape[0]
                S = int(N ** 0.5)
                if S * S == N:
                    sal_map = ga.reshape(S, S).unsqueeze(0).unsqueeze(0)  # [1,1,S,S]
                    sal_up = F.interpolate(sal_map, size=image.size[::-1], mode="bicubic", align_corners=False)[0, 0]
                    sal_np = sal_up.detach().cpu().numpy()
                    sal_np = self._percentile_normalize(sal_np, 2, 98) if enhance else self._normalize_np(sal_np)
                else:
                    sal_np = ga.detach().cpu().numpy()
                    sal_np = self._percentile_normalize(sal_np, 2, 98) if enhance else self._normalize_np(sal_np)
                # Fuse by max then min-max for consistent colors
                fused = np.maximum(self._normalize_np(cam_np), self._normalize_np(sal_np))
                cam_np = fused
            # Restore flags
            for p, f in zip(target_layer.parameters(recurse=True), orig_flags):
                p.requires_grad_(f)

            overlay = self.gradcam.overlay_on_image(
                image,
                cam_np,
                alpha=0.45,
                **({"per_pixel": True, "alpha_min": 0.0, "alpha_max": 0.6} if enhance else {})
            )

        # Save both original and overlay images
        uid = uuid.uuid4().hex[:12]
        orig_path = os.path.join(save_dir, f"orig_{uid}.png")
        cam_path = os.path.join(save_dir, f"gradcam_{uid}.png")
        image.convert("RGB").save(orig_path)
        overlay.save(cam_path)

        return {
            "pred_idx": pred_idx,
            "pred_label": self.class_names[pred_idx],
            "probs": {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))},
            "orig_path": orig_path,
            "gradcam_path": cam_path,
        }

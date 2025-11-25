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
from app.models.swin import SwinTinyFull


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
        default_pth = os.path.join(weights_dir, "best_fusion_model.pth")
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
                # Accept several common checkpoint structures
                candidate_sd = None
                if isinstance(sd, dict):
                    if any(k in sd for k in ("model", "model_state_dict", "state_dict")):
                        for k in ("model", "model_state_dict", "state_dict"):
                            if k in sd and isinstance(sd[k], dict):
                                candidate_sd = sd[k]
                                break
                    # If keys look like real param tensors (e.g. 'classifier.0.weight'), treat full dict as state_dict
                    if candidate_sd is None and all(isinstance(v, torch.Tensor) for v in sd.values()):
                        candidate_sd = sd
                # Fallback: if not dict or no match, assume it's already a state_dict
                if candidate_sd is None:
                    candidate_sd = sd
                self.model.load_state_dict(candidate_sd, strict=False)
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

        # Optional: prepare a standalone EfficientNet-B0 classifier for separate visualization
        try:
            from app.models.cnn_b0 import EffNetClassifier

            self.effnet = EffNetClassifier(num_classes=len(self.class_names), img_size=self.img_size)
            self.effnet.to(self.device).eval()
            # Try to load a matching effnet checkpoint if present
            eff_candidates = [
                os.path.join(weights_dir, "best_effnetb0.pth"),
            ]
            if os.path.isdir(weights_dir):
                for fn in os.listdir(weights_dir):
                    if fn.lower().startswith("eff") and fn.lower().endswith(".pth"):
                        p = os.path.join(weights_dir, fn)
                        if p not in eff_candidates:
                            eff_candidates.append(p)

            for p in eff_candidates:
                if os.path.exists(p):
                    try:
                        sd = torch.load(p, map_location=self.device)
                        # Attempt to load either full model or state_dict
                        try:
                            self.effnet.load_state_dict(sd)
                        except Exception:
                            # maybe checkpoint stores model key
                            if "model_state_dict" in sd:
                                self.effnet.load_state_dict(sd["model_state_dict"])
                        print(f"Loaded EffNet checkpoint: {os.path.basename(p)}")
                        break
                    except Exception as e:
                        print(f"Failed to load effnet checkpoint {p}: {e}")
        except Exception:
            self.effnet = None

        # Optional: prepare a standalone Swin-Tiny classifier (separate from fusion model)
        try:
            self.swin_cls = SwinTinyFull(num_classes=len(self.class_names))
            self.swin_cls.to(self.device).eval()
            # Try to load an explicit Swin checkpoint if present
            swin_candidates = [
                os.path.join(weights_dir, "best_swin.pth"),
            ]
            if os.path.isdir(weights_dir):
                for fn in os.listdir(weights_dir):
                    if fn.lower().startswith("best_swin") and fn.lower().endswith(".pth"):
                        p = os.path.join(weights_dir, fn)
                        if p not in swin_candidates:
                            swin_candidates.append(p)

            for p in swin_candidates:
                if os.path.exists(p):
                    try:
                        sd = torch.load(p, map_location=self.device)
                        try:
                            self.swin_cls.load_state_dict(sd, strict=False)
                        except Exception:
                            # allow nested dicts or keys under 'model_state_dict'
                            if isinstance(sd, dict) and "model_state_dict" in sd:
                                self.swin_cls.load_state_dict(sd["model_state_dict"], strict=False)
                        print(f"Loaded Swin checkpoint: {os.path.basename(p)}")
                        break
                    except Exception as e:
                        print(f"Failed to load swin checkpoint {p}: {e}")
        except Exception as e:
            print(f"[swin] standalone classifier init failed: {e}")
            self.swin_cls = None

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
        per_pixel: bool = False,
        alpha_min: float = 0.0,
        alpha_max: float = 0.6,
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
            # Match fusion's visual treatment: optionally apply percentile normalization when enhance=True
            cam_np = self._percentile_normalize(cam_np, 2, 98) if enhance else self._normalize_np(cam_np)
            overlay = self.gradcam.overlay_on_image(
                image,
                cam_np,
                alpha=0.45,
                per_pixel=per_pixel or enhance,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
        elif mode == "effnet":
            # Use standalone EfficientNet-B0 model for Grad-CAM visualization
            if getattr(self, "effnet", None) is None:
                # Explicitly fail fast so clients know EffNet visualization isn't available
                raise RuntimeError("EffNet visualization not loaded on server")

            # Compute logits for reporting
            with torch.no_grad():
                logits = self.effnet(x_eff)
                probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))

            # Compute Grad-CAM map via EffNet helper (this will run backward)
            cam_np = self.effnet.compute_gradcam_map(x_eff, target_class=pred_idx)
            # Apply same percentile normalization as fusion when requested
            cam_np = self._percentile_normalize(cam_np, 2, 98) if enhance else self._normalize_np(cam_np)
            # Overlay using the existing GradCAM utility
            overlay = self.gradcam.overlay_on_image(
                image,
                cam_np,
                alpha=0.45,
                per_pixel=per_pixel,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
        elif mode == "swin":
            # Standalone Swin-Tiny classifier with token-based saliency overlay
            if getattr(self, "swin_cls", None) is None:
                raise RuntimeError("Swin classifier not loaded on server")

            # Compute logits for reporting
            with torch.enable_grad():
                x_swin.requires_grad_(True)
                out = self.swin_cls.swin(pixel_values=x_swin, output_hidden_states=True, return_dict=True)
                logits = out.logits
                probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))

                # Use last hidden state if available
                hidden = None
                if getattr(out, "hidden_states", None):
                    hidden = out.hidden_states[-1]  # [B,N,D]
                else:
                    # Fallback: forward base model to get hidden_states
                    try:
                        base_out = self.swin_cls.swin.base_model(pixel_values=x_swin, output_hidden_states=True, return_dict=True)
                        hidden = base_out.hidden_states[-1]
                    except Exception:
                        hidden = None

                if hidden is None:
                    # If no hidden, just save original (no overlay)
                    cam_np = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
                else:
                    hidden.retain_grad()
                    score = logits[:, pred_idx].sum()
                    score.backward()
                    token_grads = hidden.grad  # [B,N,D]
                    if token_grads is not None:
                        ga = (token_grads[0] * hidden.detach()[0]).sum(dim=-1)  # [N]
                        ga = F.relu(ga)
                        N = ga.shape[0]
                        S = int(N ** 0.5)
                        if S * S == N:
                            sal_map = ga.reshape(S, S).unsqueeze(0).unsqueeze(0)
                            sal_up = F.interpolate(sal_map, size=image.size[::-1], mode="bicubic", align_corners=False)[0, 0]
                            cam_np = sal_up.detach().cpu().numpy()
                        else:
                            cam_np = ga.detach().cpu().numpy()
                        cam_np = self._percentile_normalize(cam_np, 2, 98) if enhance else self._normalize_np(cam_np)
                    else:
                        cam_np = np.zeros((image.size[1], image.size[0]), dtype=np.float32)

            overlay = self.gradcam.overlay_on_image(
                image,
                cam_np,
                alpha=0.45,
                per_pixel=per_pixel or enhance,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
        # removed: fusion_attn post-hoc cross-attention visualization (deprecated)
        elif mode == "swin_patchcam":
            # ViT token-only saliency (Patch-CAM at last stage tokens)
            # We'll support either a single selected stage or 'all' to aggregate across
            # all available Swin token stages. First compute logits once to pick pred_idx.
            with torch.enable_grad():
                ts0 = (token_stage or "hr")
                logits0, _ = self.model.forward_with_tokens(x_eff, x_swin, token_stage=ts0)
            probs = F.softmax(logits0, dim=1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))

            def _saliency_for_stage(stage_key: str):
                # returns a 2D numpy saliency upsampled to image size
                try:
                    self.model.zero_grad(set_to_none=True)
                    x_swin.requires_grad_(True)
                    with torch.enable_grad():
                        logits, swin_tokens = self.model.forward_with_tokens(x_eff, x_swin, token_stage=stage_key)
                    swin_tokens.retain_grad()
                    score = logits[:, pred_idx].sum()
                    score.backward()
                    token_grads = swin_tokens.grad  # [B,N,D]
                    if token_grads is None:
                        return None
                    ga = (token_grads[0] * swin_tokens.detach()[0]).sum(dim=-1)  # [N]
                    ga = F.relu(ga)
                    N = ga.shape[0]
                    S = int(N ** 0.5)
                    if S * S == N:
                        sal_map = ga.reshape(S, S).unsqueeze(0).unsqueeze(0)
                        sal_up = F.interpolate(sal_map, size=image.size[::-1], mode="bicubic", align_corners=False)[0, 0]
                        cam_np_local = sal_up.detach().cpu().numpy()
                    else:
                        cam_np_local = ga.detach().cpu().numpy()
                    cam_np_local = self._percentile_normalize(cam_np_local, 2, 98) if enhance else self._normalize_np(cam_np_local)
                    return cam_np_local
                except Exception:
                    return None

            # If user requested all stages, enumerate available hidden state sizes
            stages_to_check = []
            if (token_stage or "") == "all":
                try:
                    swin_out = self.model.swin(pixel_values=x_swin, output_hidden_states=True, return_dict=True)
                    sizes = set()
                    if swin_out.hidden_states is not None:
                        for hs in swin_out.hidden_states:
                            n = hs.shape[1]
                            s = int(n ** 0.5)
                            if s * s == n:
                                sizes.add(s)
                    # Always include global pooled token
                    stages_to_check = ["1"] + sorted([str(s) for s in sizes])
                except Exception:
                    stages_to_check = ["7"]
            else:
                stages_to_check = [str(token_stage or "7")]

            maps = []
            for st in stages_to_check:
                m = _saliency_for_stage(st)
                if m is not None:
                    maps.append(m)

            if not maps:
                cam_np = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
            else:
                # fuse by elementwise max for visibility consistency
                cam_np = maps[0]
                for m in maps[1:]:
                    cam_np = np.maximum(cam_np, m)

            overlay = self.gradcam.overlay_on_image(
                image,
                cam_np,
                alpha=0.45,
                per_pixel=per_pixel or enhance,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
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
                per_pixel=per_pixel or enhance,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
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

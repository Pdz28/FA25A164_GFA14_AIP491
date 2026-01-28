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

from app.models.hybridmodel import CNNViTFusion
from app.utils.gradcam import GradCAM
from app.models.swintiny import SwinTinyFull


class InferenceService:
    def __init__(self, checkpoints_dir: str, device: str | torch.device = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = ["benign", "malignant"]
        self.img_size = (224, 224)
        self.loaded_checkpoints_info = ""
        self._build_transforms()
        self._load_model(checkpoints_dir)

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

    def _load_model(self, checkpoints_dir: str):
        from peft import get_peft_model, LoraConfig
        
        # Initialize base model with new architecture parameters
        self.model = CNNViTFusion(
            num_classes=len(self.class_names), 
            img_size=self.img_size,
            d_model=768,
            num_heads=8
        )
        self.model.to(self.device).eval()
        
        # Apply LoRA to Swin Transformer
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "q_proj", "k_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
            modules_to_save=[],
        )
        self.model.swin = get_peft_model(self.model.swin, lora_config)

        # 1) Try full state dict (.pth) - prioritize hybrid_model.pth
        pth_candidates = []
        hybrid_pth = os.path.join(checkpoints_dir, "hybrid_model.pth")
        if os.path.exists(hybrid_pth):
            pth_candidates.append(hybrid_pth)
        
        # default_pth = os.path.join(checkpoints_dir, "best_fusion_model.pth")
        # if os.path.exists(default_pth):
        #     pth_candidates.append(default_pth)
        
        # any .pth in checkpoints_dir
        if os.path.isdir(checkpoints_dir):
            for fn in os.listdir(checkpoints_dir):
                if fn.lower().endswith(".pth"):
                    path = os.path.join(checkpoints_dir, fn)
                    if path not in pth_candidates:
                        pth_candidates.append(path)

        loaded = False
        for path in pth_candidates:
            try:
                sd = torch.load(path, map_location=self.device)
                # Accept several common checkpoint structures
                candidate_sd = None
                if isinstance(sd, dict):
                    if any(k in sd for k in ("model", "model_state_dict", "state_dict", "model_state")):
                        for k in ("model", "model_state_dict", "state_dict", "model_state"):
                            if k in sd and isinstance(sd[k], dict):
                                candidate_sd = sd[k]
                                break
                    # If keys look like real param tensors (e.g. 'classifier.0.weight'), treat full dict as state_dict
                    if candidate_sd is None and all(isinstance(v, torch.Tensor) for v in sd.values()):
                        candidate_sd = sd
                # Fallback: if not dict or no match, assume it's already a state_dict
                if candidate_sd is None:
                    candidate_sd = sd
                
                load_result = self.model.load_state_dict(candidate_sd, strict=False)
                self.loaded_checkpoints_info = f"Loaded checkpoint: {os.path.basename(path)}"
                if load_result.missing_keys:
                    print(f"Missing keys: {len(load_result.missing_keys)}")
                if load_result.unexpected_keys:
                    print(f"Unexpected keys: {len(load_result.unexpected_keys)}")
                print(self.loaded_checkpoints_info)
                loaded = True
                break
            except Exception as e:
                print(f"Failed to load {path}: {e}")

        # 2) If no full .pth, try LoRA adapter directory
        if not loaded and os.path.isdir(checkpoints_dir):
            adapter_dirs = []
            for name in os.listdir(checkpoints_dir):
                ap = os.path.join(checkpoints_dir, name)
                if os.path.isdir(ap) and (
                    os.path.exists(os.path.join(ap, "adapter_config.json"))
                    or os.path.exists(os.path.join(ap, "adapter_model.bin"))
                ):
                    adapter_dirs.append(ap)

            for ap in adapter_dirs:
                try:
                    # self.model.swin is a PEFT-wrapped model; load adapter checkpoints
                    self.model.swin.load_adapter(ap, adapter_name="default")
                    self.model.swin.set_adapter("default")
                    self.loaded_checkpoints_info = f"Loaded LoRA adapter: {os.path.basename(ap)}"
                    print(self.loaded_checkpoints_info)
                    loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load adapter {ap}: {e}")

        if not loaded:
            self.loaded_checkpoints_info = "No trained checkpoints found; using backbone defaults with CrossModelAttention fusion."
            print(self.loaded_checkpoints_info)

        # Prepare Grad-CAM on the last CNN block
        target_layer = self.model.get_last_cnn_layer()
        self.gradcam = GradCAM(self.model, target_layer)

        # Optional: prepare a standalone EfficientNet-B0 classifier for separate visualization
        try:
            from app.models.efficientnetb0 import EffNetClassifier

            self.effnet = EffNetClassifier(num_classes=len(self.class_names), img_size=self.img_size)
            self.effnet.to(self.device).eval()
            # Try to load a matching effnet checkpoint if present
            eff_candidates = [
                os.path.join(checkpoints_dir, "best_effnetb0.pth"),
            ]
            if os.path.isdir(checkpoints_dir):
                for fn in os.listdir(checkpoints_dir):
                    if fn.lower().startswith("eff") and fn.lower().endswith(".pth"):
                        p = os.path.join(checkpoints_dir, fn)
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
                os.path.join(checkpoints_dir, "swin-tiny"),
            ]
            if os.path.isdir(checkpoints_dir):
                for fn in os.listdir(checkpoints_dir):
                    if fn.lower().startswith("swin-tiny") and fn.lower().endswith(".pth"):
                        p = os.path.join(checkpoints_dir, fn)
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
        bn_mode: str = "running",        # "running" (eval) or "batch" (force batch stats)
        dropout_p: float | None = None,  # override dropout prob (e.g., 0.0 for off)
        fixed_w_cnn: float | None = None,# optional fixed fusion weight for CNN map
        fixed_w_vit: float | None = None,# optional fixed fusion weight for ViT map
        ) -> Dict:
        os.makedirs(save_dir, exist_ok=True)
        # Global eval for inference stability
        self.model.eval()
        if getattr(self, "effnet", None) is not None:
            self.effnet.eval()
        if getattr(self, "swin_cls", None) is not None:
            self.swin_cls.eval()

        # Apply BN mode: "running" uses eval (default), "batch" forces batch stats during forward
        def _set_bn_mode(module: torch.nn.Module, use_batch_stats: bool):
            for m in module.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    # Toggle BN to train to use batch stats, but freeze affine params
                    if use_batch_stats:
                        m.train()
                    else:
                        m.eval()

        _set_bn_mode(self.model, bn_mode == "batch")
        if getattr(self, "effnet", None) is not None:
            _set_bn_mode(self.effnet, bn_mode == "batch")

        # Apply dropout override if requested
        def _set_dropout_p(module: torch.nn.Module, pval: float | None):
            if pval is None:
                return
            for m in module.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = float(pval)

        _set_dropout_p(self.model, dropout_p)
        if getattr(self, "effnet", None) is not None:
            _set_dropout_p(self.effnet, dropout_p)
        if getattr(self, "swin_cls", None) is not None:
            _set_dropout_p(self.swin_cls, dropout_p)
        x_eff, x_swin = self.preprocess(image)

        if mode == "cnn":
            # Old behavior: CAM from the CNN branch only (still post-fusion w.r.t. logit)
            logits, alpha, cnn_logits, swin_logits = self.model(x_eff, x_swin)
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
            # Dùng pytorch-grad-cam cho nhánh EfficientNetB0 thuần
            if getattr(self, "effnet", None) is None:
                raise RuntimeError("EffNet visualization not loaded on server")

            # Tính logits để xác định lớp mục tiêu
            with torch.no_grad():
                logits = self.effnet(x_eff)
                probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))

            # Sử dụng thư viện grad-cam chính thức
            try:
                from pytorch_grad_cam import GradCAM as PTGradCAM
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                # Tránh lỗi __del__ khi đối tượng chưa khởi tạo đầy đủ
                try:
                    PTGradCAM.__del__ = lambda self: None  # monkey-patch safe destructor
                except Exception:
                    pass
            except Exception as e:
                raise RuntimeError(f"pytorch-grad-cam not available: {e}")

            target_layer = self.effnet.get_last_cnn_layer()
            if target_layer is None:
                raise RuntimeError("EffNet last conv layer not found")
            target_layers = [target_layer]
            # Khởi tạo GradCAM theo chữ ký tương thích, để lib tự suy luận device
            cam_extractor = PTGradCAM(self.effnet, target_layers)
            try:
                targets = [ClassifierOutputTarget(pred_idx)]
                grayscale_cam = cam_extractor(input_tensor=x_eff, targets=targets)
                if isinstance(grayscale_cam, np.ndarray):
                    cam_np = grayscale_cam[0]
                else:
                    cam_np = grayscale_cam[0].detach().cpu().numpy()
            finally:
                # Giải phóng hook nếu lib hỗ trợ
                try:
                    cam_extractor.activations_and_grads.release()
                except Exception:
                    pass

            # Chuẩn hoá và ghép ảnh
            cam_np = self._percentile_normalize(cam_np, 2, 98) if enhance else self._normalize_np(cam_np)
            overlay = self.gradcam.overlay_on_image(
                image,
                cam_np,
                alpha=0.45,
                per_pixel=per_pixel or enhance,
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
                logits0, swin_tokens0, cnn_pool0, swin_pool0 = self.model.forward_with_tokens(x_eff, x_swin, token_stage=ts0)
            probs = F.softmax(logits0, dim=1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            def _saliency_for_stage(stage_key: str):
                # returns a 2D numpy saliency upsampled to image size
                try:
                    self.model.zero_grad(set_to_none=True)
                    x_swin.requires_grad_(True)
                    with torch.enable_grad():
                        logits, swin_tokens, cnn_pool_stage, swin_pool_stage = self.model.forward_with_tokens(x_eff, x_swin, token_stage=stage_key)
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
                logits, swin_tokens, pooled_cnn, pooled_swin = self.model.forward_with_tokens(x_eff, x_swin, token_stage=ts)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))

            # retain grad for tokens
            swin_tokens.retain_grad()
            # retain grads for branch pooled features to compute checkpoints
            try:
                pooled_cnn.retain_grad()
                pooled_swin.retain_grad()
            except Exception:
                pass
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
                else:
                    sal_np = ga.detach().cpu().numpy()
                sal_np = self._percentile_normalize(sal_np, 2, 98) if enhance else self._normalize_np(sal_np)
            else:
                sal_np = np.zeros((image.size[1], image.size[0]), dtype=np.float32)

            # Gradient-weighted branch fusion
            try:
                pooled_cnn.retain_grad()
                pooled_swin.retain_grad()
            except Exception:
                pass
            g_cnn = pooled_cnn.grad.detach().abs().mean().item() if getattr(pooled_cnn, 'grad', None) is not None else 0.0
            g_vit = pooled_swin.grad.detach().abs().mean().item() if getattr(pooled_swin, 'grad', None) is not None else 0.0
            if fixed_w_cnn is not None and fixed_w_vit is not None:
    # Use provided fixed weights; normalize to sum=1 for consistency
                total = fixed_w_cnn + fixed_w_vit
                if total <= 1e-12:
                    w_cnn, w_vit = 0.5, 0.5
                else:
                    w_cnn = float(fixed_w_cnn) / total
                    w_vit = float(fixed_w_vit) / total
            else:
                denom = (g_cnn + g_vit) if (g_cnn + g_vit) > 1e-12 else 1.0
                w_cnn = g_cnn / denom
                w_vit = g_vit / denom

            cam_np_n = self._normalize_np(cam_np)
            sal_np_n = self._normalize_np(sal_np)
            cam_np = self._normalize_np(w_cnn * cam_np_n + w_vit * sal_np_n)
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
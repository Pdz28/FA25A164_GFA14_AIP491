from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


def apply_colormap(cam: np.ndarray, colormap: str = "jet", gamma: float = 0.9) -> np.ndarray:
    """Map a [0,1] float CAM to an RGB heatmap.

    - jet: classic blue→cyan→yellow→red used in many Grad‑CAM figures.
    No external dependency required.
    """
    v = np.clip(cam, 0.0, 1.0).astype(np.float32)
    # Optional gamma to increase contrast toward the hot regions
    if gamma and gamma != 1.0:
        v = np.power(v, gamma)

    if colormap.lower() == "jet":
        # Piecewise triangular JET approximation in [0,1]
        # r,g,b peaks around 3/4, 2/4, 1/4 respectively
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    else:
        # Fallback: grayscale to red
        r, g, b = v, v * 0.4, np.zeros_like(v)

    heatmap = np.stack([r, g, b], axis=-1)
    heatmap = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)
    return heatmap


class GradCAM:
    """Grad-CAM for the CNN branch of the fusion model.

    Hooks the last conv block of EfficientNet features to produce a heatmap for a chosen class.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            # grad_out is a tuple with gradients w.r.t. the module outputs
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    @torch.no_grad()
    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        t_min, t_max = t.min(), t.max()
        if (t_max - t_min) < 1e-6:
            return torch.zeros_like(t)
        return (t - t_min) / (t_max - t_min)

    def generate(
        self,
        x_cnn: torch.Tensor,
        x_swin: torch.Tensor,
        target_class: Optional[int] = None,
        upsample_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Return CAM heatmap tensor in [0,1] shape [H,W].

        Note: If the CNN backbone is frozen (requires_grad=False), we temporarily
        enable gradients on the target layer so that backward hooks receive grads.
        """

        def _toggle_requires_grad(module: torch.nn.Module, flag: bool):
            for p in module.parameters(recurse=True):
                p.requires_grad_(flag)

        # Temporarily ensure target layer participates in autograd
        orig_flags = [p.requires_grad for p in self.target_layer.parameters(recurse=True)]
        _toggle_requires_grad(self.target_layer, True)

        self.model.zero_grad(set_to_none=True)

        # Make sure grad tracking is enabled during forward
        with torch.enable_grad():
            logits = self.model(x_cnn, x_swin)  # forward collects activations
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        score = logits[:, target_class].sum()
        score.backward()

        # activations: [B,C,H,W], gradients: [B,C,H,W]
        acts = self.activations
        grads = self.gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        cam = self._normalize(cam)
        if upsample_size is not None:
            cam = F.interpolate(cam, size=upsample_size, mode="bilinear", align_corners=False)
        out = cam[0, 0].detach().cpu()

        # Restore original requires_grad flags on target layer
        for p, f in zip(self.target_layer.parameters(recurse=True), orig_flags):
            p.requires_grad_(f)

        return out

    @staticmethod
    def overlay_on_image(
        pil_image: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.45,
        per_pixel: bool = False,
        alpha_min: float = 0.0,
        alpha_max: float = 0.6,
        colormap: str = "jet",
        smooth_mask: bool = True,
    ) -> Image.Image:
        """Overlay a heatmap on top of the input image.

        - If per_pixel=False: classic constant-alpha blend.
        - If per_pixel=True: use the normalized CAM as an alpha map in [alpha_min, alpha_max]
          for spatially varying transparency; optionally smooth the mask to reduce blockiness.
        """
        base = pil_image.convert("RGB")
        v = np.clip(cam, 0.0, 1.0).astype(np.float32)
        heatmap = apply_colormap(v, colormap=colormap)
        heat = Image.fromarray(heatmap).resize(base.size, Image.BICUBIC)

        if not per_pixel:
            return Image.blend(base, heat, alpha)

        # Per-pixel alpha compositing
        # Build mask in 0..255 from [alpha_min, alpha_max] scaled by CAM
        mask_f = (alpha_min + (alpha_max - alpha_min) * v)
        mask = Image.fromarray((np.clip(mask_f, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
            base.size, Image.BICUBIC
        )
        if smooth_mask:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=0.5))

        heat_rgba = heat.convert("RGBA")
        heat_rgba.putalpha(mask)
        base_rgba = base.convert("RGBA")
        out = Image.alpha_composite(base_rgba, heat_rgba).convert("RGB")
        return out

    @staticmethod
    def to_png_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def build_from_cached(self, upsample_size: Tuple[int, int] | None) -> torch.Tensor:
        """Build CAM from cached activations/gradients captured by hooks.

        Use this after you've already run a forward + backward elsewhere.
        Returns a tensor [H,W] in [0,1].
        """
        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError("GradCAM cache is empty; run forward+backward first.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = self._normalize(cam)
        if upsample_size is not None:
            cam = F.interpolate(cam, size=upsample_size, mode="bilinear", align_corners=False)
        return cam[0, 0].detach().cpu()

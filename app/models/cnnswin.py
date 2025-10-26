import torch
import torch.nn as nn
from torchvision import models
from transformers import SwinModel
from peft import get_peft_model, LoraConfig


class AttentionFusion(nn.Module):
    def __init__(self, d_cnn: int, d_swin: int, d_model: int = 768, num_heads: int = 12, dropout: float = 0.2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model

        self.cnn_proj = nn.Linear(d_cnn, d_model)
        self.swin_proj = nn.Linear(d_swin, d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_out1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_out2 = nn.LayerNorm(d_model)

    def forward(self, cnn_feat: torch.Tensor, swin_feat: torch.Tensor) -> torch.Tensor:
        q_proj = self.norm_q(self.swin_proj(swin_feat))
        kv_proj = self.norm_kv(self.cnn_proj(cnn_feat))

        q = q_proj.unsqueeze(1)  # [B,1,D]
        kv = kv_proj.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        attn_out = attn_out.squeeze(1)

        fused = self.norm_out1(attn_out + q_proj + kv_proj)
        fused = fused + self.ffn(fused)
        fused = self.norm_out2(fused)
        return fused


class CNNViTFusion(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        img_size=(224, 224),
        lora_r: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.05,
        fusion_d_model: int = 768,
        fusion_num_heads: int = 12,
        freeze_cnn: bool = True,
        freeze_swin_except_lora: bool = True,
    ):
        super().__init__()
        # CNN backbone (EfficientNet-B3)
        effnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(*list(effnet.features.children()))
        self.cnn_bn = nn.BatchNorm2d(1536)
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size[0], img_size[1])
            feat = self.cnn_backbone(dummy)
            pooled = self.cnn_pool(feat).view(1, -1)
            self.cnn_out_dim = pooled.shape[1]

        # Swin + LoRA (Tiny)
        swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=[
                "query",
                "key",
                "value",
                "attention.output.dense",
                "mlp.dense1",
                "mlp.dense2",
            ],
        )
        self.swin = get_peft_model(swin, lora_config)
        self.swin_hidden = self.swin.config.hidden_size

        # Fusion + classifier
        self.fuse = AttentionFusion(
            d_cnn=self.cnn_out_dim,
            d_swin=self.swin_hidden,
            d_model=fusion_d_model,
            num_heads=fusion_num_heads,
            dropout=0.2,
        )

        fused_dim = self.fuse.d_model
        self.pre_classifier_norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        if freeze_cnn:
            for p in self.cnn_backbone.parameters():
                p.requires_grad = False
        if freeze_swin_except_lora:
            for n, p in self.swin.named_parameters():
                p.requires_grad = False
            for n, p in self.swin.named_parameters():
                if "lora" in n.lower() or "adapter" in n.lower() or "alpha" in n.lower():
                    p.requires_grad = True

    def forward(self, x_cnn: torch.Tensor, x_swin: torch.Tensor) -> torch.Tensor:
        feats = self.cnn_backbone(x_cnn)  # [B,1536,H,W]
        feats = self.cnn_bn(feats)
        pooled = self.cnn_pool(feats).view(feats.size(0), -1)

        swin_out = self.swin(pixel_values=x_swin)
        swin_feats = swin_out.last_hidden_state.mean(dim=1)

        fused = self.fuse(pooled, swin_feats)
        fused = self.pre_classifier_norm(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

    def forward_with_tokens(self, x_cnn: torch.Tensor, x_swin: torch.Tensor, token_stage: str = "last"):
        """Forward that also returns Swin tokens for visualization.

        Args:
            token_stage: "last" (default, 7x7)
                         "hr" (prefer 14x14 if available)
                         "14", "28", or "7" to explicitly pick the grid size

        Returns:
            logits, swin_tokens (selected by token_stage)
        """
        feats = self.cnn_backbone(x_cnn)
        feats = self.cnn_bn(feats)
        pooled = self.cnn_pool(feats).view(feats.size(0), -1)

        # Ask Swin for hidden states so we can choose a higher‑resolution token grid
        swin_out = self.swin(pixel_values=x_swin, output_hidden_states=True, return_dict=True)
        last_tokens = swin_out.last_hidden_state  # [B,N,D]

        selected_tokens = last_tokens
        token_stage_l = (token_stage or "last").lower()
        if swin_out.hidden_states is not None:
            # helper to compute square grid size
            # pick the first hidden_state whose token grid is larger than the last (e.g., 14x14)
            try:
                import math

                def grid_size(t):
                    n = t.shape[1]
                    s = int(math.sqrt(n))
                    return s if s * s == n else None

                last_s = grid_size(last_tokens) or 7
                if token_stage_l in ("56", "28", "14", "7"):
                    target = int(token_stage_l)
                    for hs in swin_out.hidden_states:
                        s = grid_size(hs)
                        if s == target:
                            selected_tokens = hs
                            break
                elif token_stage_l in ("hr", "stage3", "high", "highres"):
                    candidates = []
                    for hs in swin_out.hidden_states:
                        s = grid_size(hs)
                        if s is not None and s > last_s:
                            candidates.append((s, hs))
                    chosen = None
                    for s, hs in candidates:
                        if s == 14:
                            chosen = hs
                            break
                    if chosen is None and candidates:
                        candidates.sort(key=lambda x: x[0])
                        chosen = candidates[0][1]
                    if chosen is not None:
                        selected_tokens = chosen
            except Exception:
                selected_tokens = last_tokens

        swin_feats = last_tokens.mean(dim=1)  # keep classifier behavior identical

        fused = self.fuse(pooled, swin_feats)
        fused = self.pre_classifier_norm(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits, selected_tokens

    # Helper to get a reference to the last conv layer for Grad-CAM hooks
    def get_last_cnn_layer(self) -> nn.Module:
        return list(self.cnn_backbone.children())[-1]

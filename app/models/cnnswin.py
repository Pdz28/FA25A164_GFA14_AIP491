import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from transformers import SwinModel
from peft import get_peft_model, LoraConfig


# class AttentionFusion(nn.Module):
#     def __init__(self, d_cnn: int, d_swin: int, d_model: int = 768, num_heads: int = 12, dropout: float = 0.2):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
#         self.d_model = d_model

#         self.cnn_proj = nn.Linear(d_cnn, d_model)
#         self.swin_proj = nn.Linear(d_swin, d_model)
#         self.norm_q = nn.LayerNorm(d_model)
#         self.norm_kv = nn.LayerNorm(d_model)

#         self.attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.norm_out1 = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model),
#         )
#         self.norm_out2 = nn.LayerNorm(d_model)

#     def forward(self, cnn_feat: torch.Tensor, swin_feat: torch.Tensor) -> torch.Tensor:
#         q_proj = self.norm_q(self.swin_proj(swin_feat))
#         kv_proj = self.norm_kv(self.cnn_proj(cnn_feat))

#         q = q_proj.unsqueeze(1)  # [B,1,D]
#         kv = kv_proj.unsqueeze(1)
#         attn_out, _ = self.attn(q, kv, kv)
#         attn_out = attn_out.squeeze(1)

#         fused = self.norm_out1(attn_out + q_proj + kv_proj)
#         fused = fused + self.ffn(fused)
#         fused = self.norm_out2(fused)
#         return fused


class CNNViTFusion(nn.Module):
    def __init__(self, num_classes=2, img_size=(224,224)):
        super().__init__()
        # 1. CNN Backbone (EfficientNet)
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(*list(effnet.features.children()))
        self.cnn_bn = nn.BatchNorm2d(1280)
        self.cnn_pool = nn.AdaptiveAvgPool2d((1,1))
        self.cnn_out_bn1d = nn.BatchNorm1d(1280)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size[0], img_size[1])
            feat = self.cnn_backbone(dummy)
            pooled = self.cnn_pool(feat).view(1, -1)
            self.cnn_out_dim = pooled.shape[1]

        # 2. Swin Transformer
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.swin_hidden = self.swin.config.hidden_size
        self.swin_out_norm = nn.LayerNorm(self.swin_hidden)
        
        # 3. Fusion Head
        fused_dim = self.cnn_out_dim + self.swin_hidden
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
           nn.Linear(fused_dim, 512),
           nn.SiLU(),
           nn.Dropout(0.2),
           nn.Linear(512, 256),
           nn.SiLU(),
           nn.Dropout(0.1),
           nn.Linear(256, num_classes)
        )

        # --- STRATEGY SETUP ---
        # A. Freeze CNN Backbone ban đầu
        for p in self.cnn_backbone.parameters():
            p.requires_grad = False
            
        # B. Unfreeze Swin NGAY TỪ ĐẦU (Train Full)
        for n, p in self.swin.named_parameters():
            p.requires_grad = True

    def forward(self, x_cnn, x_swin):
        # CNN
        feats = self.cnn_backbone(x_cnn)
        feats = self.cnn_bn(feats)
        pooled = self.cnn_pool(feats).view(feats.size(0), -1)
        pooled = self.cnn_out_bn1d(pooled)

        # Swin
        swin_out = self.swin(pixel_values=x_swin)
        swin_feats = swin_out.last_hidden_state.mean(dim=1)
        swin_feats = self.swin_out_norm(swin_feats)

        # Fusion
        fused = torch.cat((pooled, swin_feats), dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

    def forward_with_tokens(self, x_cnn: torch.Tensor, x_swin: torch.Tensor, token_stage: str = "7"):
        """Forward that also returns Swin tokens for visualization.

        Args:
            token_stage: "7" (default, 7x7)
                         "hr" (prefer 14x14 if available)
                         "14", "28", "7" or "1" to explicitly pick the grid size

        Returns:
            logits, swin_tokens (selected or resampled to token_stage)
        """
        # CNN branch (mirror main forward, including BatchNorm1d)
        feats = self.cnn_backbone(x_cnn)
        feats = self.cnn_bn(feats)
        pooled = self.cnn_pool(feats).view(feats.size(0), -1)
        pooled = self.cnn_out_bn1d(pooled)

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

        # If caller requested a specific token stage, try to return tokens matching that size.
        # Support token_stage == "1" (global pooled token) and numeric sizes via adaptive pooling.
        try:
            if token_stage_l in ("1", "global", "pool"):
                # global pooled token [B,1,D]
                selected_tokens = last_tokens.mean(dim=1, keepdim=True)
            elif token_stage_l in ("56", "28", "14", "7"):
                target = int(token_stage_l)
                B, N, D = selected_tokens.shape
                cur_s = int(N ** 0.5) if int(N ** 0.5) ** 2 == N else None
                if cur_s is not None and cur_s != target:
                    # reshape to (B, D, H, W) for pooling
                    toks = selected_tokens.view(B, cur_s, cur_s, D).permute(0, 3, 1, 2).contiguous()
                    toks_p = F.adaptive_avg_pool2d(toks, (target, target))
                    selected_tokens = toks_p.permute(0, 2, 3, 1).contiguous().view(B, target * target, D)
        except Exception:
            # safe fallback to whatever we had
            pass

        # Aggregate Swin features same as main forward
        swin_feats = last_tokens.mean(dim=1)
        swin_feats = self.swin_out_norm(swin_feats)

        # Simple concatenation fusion (old self.fuse + norm removed)
        fused = torch.cat((pooled, swin_feats), dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits, selected_tokens

    # Helper to get a reference to the last conv layer for Grad-CAM hooks
    def get_last_cnn_layer(self) -> nn.Module:
        return list(self.cnn_backbone.children())[-1]

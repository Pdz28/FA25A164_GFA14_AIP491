import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from transformers import SwinModel
from peft import get_peft_model, LoraConfig
import numpy as np


class AttentionFusion(nn.Module):
    """
    New advanced fusion module with:
    - Spatial token-based attention (processes CNN feature maps as tokens)
    - Learnable positional embeddings
    - Gated residual connections with dynamic alpha
    - Layer scaling for stable training
    """
    def __init__(self, cnn_channels, swin_hidden, d_model=768, num_heads=8, dropout=0.1, use_pos=True, debug=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.use_pos = use_pos
        self.debug_mode = debug
        
        # Input normalization and projection
        self.cnn_proj = nn.Linear(cnn_channels, d_model)
        self.norm_cnn_in = nn.LayerNorm(cnn_channels)
        self.swin_proj = nn.Linear(swin_hidden, d_model)
        self.norm_swin_in = nn.LayerNorm(swin_hidden)
        
        # Pre-attention normalization
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        
        # Learnable positional embeddings (7x7 = 49 tokens default)
        if self.use_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, 49, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer("pos_embed", None)
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Layer scaling for stable deep network training
        self.layerscale_attn = nn.Parameter(1e-4 * torch.ones(d_model))
        self.layerscale_ffn = nn.Parameter(1e-4 * torch.ones(d_model))
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Dynamic gating mechanism
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1)
        )
        
        self.last_debug = None

    def forward(self, cnn_feat_map, swin_tokens, debug: bool = False):
        """
        Args:
            cnn_feat_map: [B, C, H, W] - CNN spatial features
            swin_tokens: [B, N, D] - Swin transformer tokens
            debug: Whether to store debug information
            
        Returns:
            pooled: [B, d_model] - Fused features
            cnn_pool_mean: [B, d_model] - CNN pooled features (for auxiliary head)
            swin_pool_mean: [B, d_model] - Swin pooled features (for auxiliary head)
            alpha_scalar: [B, 1] - Dynamic gate values
            debug_dict: Debug information if requested
        """
        device_local = cnn_feat_map.device
        B = cnn_feat_map.size(0)
        S_swin = swin_tokens.size(1)
        
        # Resize CNN feature map to match Swin token grid
        target_h = int(max(2, round(np.sqrt(S_swin))))
        target_w = int(max(2, int(np.ceil(S_swin / float(target_h)))))
        cnn_resized = F.interpolate(cnn_feat_map, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Convert CNN spatial features to tokens
        _, C, H, W = cnn_resized.shape
        cnn_tokens = cnn_resized.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Normalize and project CNN tokens
        cnn_tokens = self.norm_cnn_in(cnn_tokens)
        cnn_proj = self.cnn_proj(cnn_tokens)  # [B, H*W, d_model]
        
        # Add positional embeddings to CNN tokens
        if self.use_pos and self.pos_embed is not None:
            L_pe = self.pos_embed.size(1)
            if L_pe == cnn_proj.size(1):
                pos = self.pos_embed.to(device_local)
            else:
                # Interpolate positional embeddings if size mismatch
                d = self.pos_embed.size(2)
                h_pe = int(round(np.sqrt(L_pe)) if L_pe > 0 else 0)
                w_pe = int(int(np.ceil(L_pe / float(h_pe))) if h_pe > 0 and L_pe > 0 else 0)
                pos_grid = self.pos_embed[0]
                
                if h_pe * w_pe != L_pe:
                    pad_len = h_pe * w_pe - L_pe
                    pad = pos_grid.new_zeros((pad_len, d))
                    pos_grid = torch.cat([pos_grid, pad], dim=0)
                
                if h_pe > 0 and w_pe > 0:
                    pos_grid = pos_grid.view(1, h_pe, w_pe, d).permute(0, 3, 1, 2)
                    pos_interp = F.interpolate(pos_grid.to(device_local), size=(target_h, target_w), mode='bilinear', align_corners=False)
                    pos = pos_interp.permute(0, 2, 3, 1).view(1, target_h * target_w, d)
                else:
                    pos = self.pos_embed.new_zeros((1, cnn_proj.size(1), d))
            
            if pos.size(1) == cnn_proj.size(1):
                cnn_proj = cnn_proj + pos
        
        # Process Swin tokens
        swin_normed = self.norm_swin_in(swin_tokens)
        swin_proj = self.swin_proj(swin_normed)  # [B, N, d_model]
        
        # Match sequence lengths if needed
        S_cnn = cnn_proj.size(1)
        if swin_proj.size(1) != S_cnn:
            idx = torch.linspace(0, swin_proj.size(1) - 1, steps=S_cnn).long().to(swin_proj.device)
            swin_proj = swin_proj[:, idx, :]
        
        # Cross-modal attention (Swin queries, CNN keys/values)
        q = self.norm_q(swin_proj)
        kv = self.norm_kv(cnn_proj)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        
        # Apply layer scaling and residual connection
        ls_attn = self.layerscale_attn.view(1, 1, -1).to(device_local)
        ls_ffn = self.layerscale_ffn.view(1, 1, -1).to(device_local)
        x_attn_res = swin_proj + ls_attn * attn_out
        
        # Dynamic gating: compute alpha from pooled features
        cnn_pool_mean = cnn_proj.mean(dim=1)
        cnn_pool_max = cnn_proj.max(dim=1).values
        swin_pool_mean = swin_proj.mean(dim=1)
        swin_pool_max = swin_proj.max(dim=1).values
        
        gate_in = torch.cat([cnn_pool_mean, cnn_pool_max, swin_pool_mean, swin_pool_max], dim=1)
        gate_logits = self.gate_mlp(gate_in)
        alpha_scalar = torch.sigmoid(gate_logits).view(-1, 1)
        alpha_b = alpha_scalar.unsqueeze(1)
        
        # Gated residual: blend attention output with original Swin features
        x = alpha_b * x_attn_res + (1.0 - alpha_b) * swin_proj
        
        # Feed-forward with residual and layer scaling
        x = x + ls_ffn * self.ffn(self.norm_ffn(x))
        
        # Global pooling
        pooled = x.mean(dim=1)
        
        # Store debug info
        self.last_debug = {"alpha": alpha_scalar.detach().cpu().numpy()}
        
        return pooled, cnn_pool_mean, swin_pool_mean, alpha_scalar, self.last_debug



class CNNViTFusion(nn.Module):
    """
    Hybrid CNN-ViT Fusion Model with:
    - EfficientNet-B0 as CNN backbone
    - Swin Transformer (with LoRA for efficient fine-tuning)
    - Advanced AttentionFusion module
    - Learnable input scaling/shifting
    - Multi-head prediction (fusion + CNN auxiliary + Swin auxiliary)
    """
    def __init__(self, num_classes=2, img_size=(224,224), d_model=768, num_heads=8):
        super().__init__()
        
        # 1. CNN Backbone (EfficientNet-B0)
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(*list(effnet.features.children()))
        self.cnn_out_channels = 1280
        
        # Learnable input preprocessing
        self.input_scale_eff = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.input_shift_eff = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.input_scale_swin = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.input_shift_swin = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        # 2. Swin Transformer
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        swin_hidden = self.swin.config.hidden_size  # 768
        
        # 3. Advanced Fusion Module
        self.fusion = AttentionFusion(
            cnn_channels=self.cnn_out_channels,
            swin_hidden=swin_hidden,
            d_model=d_model,
            num_heads=num_heads,
            dropout=0.15
        )
        
        # 4. Classification Heads
        # Auxiliary heads for CNN and Swin branches
        self.cnn_head = nn.Linear(d_model, num_classes)
        self.swin_head = nn.Linear(d_model, num_classes)
        
        # Main fusion classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
        
        # Initialize with frozen backbones (will be unfrozen during training)
        for p in self.cnn_backbone.parameters():
            p.requires_grad = False
        for p in self.swin.parameters():
            p.requires_grad = False

    def forward(self, x_cnn, x_swin, debug=False):
        """
        Args:
            x_cnn: [B, 3, H, W] - Input for CNN branch
            x_swin: [B, 3, H, W] - Input for Swin branch
            debug: Whether to return debug information
            
        Returns:
            logits: [B, num_classes] - Main fusion predictions
            alpha: [B, 1] - Dynamic gate values from fusion
            cnn_logits: [B, num_classes] - CNN auxiliary predictions
            swin_logits: [B, num_classes] - Swin auxiliary predictions
        """
        # Apply learnable input preprocessing
        x_cnn = x_cnn * self.input_scale_eff + self.input_shift_eff
        x_swin = x_swin * self.input_scale_swin + self.input_shift_swin
        
        # CNN branch: Extract spatial features
        feats = self.cnn_backbone(x_cnn)  # [B, 1280, H, W]
        
        # Swin branch: Extract transformer tokens
        swin_out = self.swin(pixel_values=x_swin, return_dict=True)
        swin_tokens = swin_out.last_hidden_state  # [B, N, 768]
        
        # Fusion: Combine CNN spatial features with Swin tokens
        fused, cnn_pool, swin_pool, alpha, dbg = self.fusion(feats, swin_tokens, debug=debug)
        
        # Generate predictions from all heads
        cnn_logits = self.cnn_head(cnn_pool)
        swin_logits = self.swin_head(swin_pool)
        logits = self.classifier(fused)
        
        return logits, alpha, cnn_logits, swin_logits

    def forward_with_tokens(self, x_cnn: torch.Tensor, x_swin: torch.Tensor, token_stage: str = "7"):
        """
        Forward pass that also returns intermediate tokens for visualization (Grad-CAM).
        
        Args:
            x_cnn: [B, 3, H, W] - Input for CNN branch
            x_swin: [B, 3, H, W] - Input for Swin branch
            token_stage: Resolution of tokens to return ("7", "14", "28", "hr")
            
        Returns:
            logits: [B, num_classes] - Main predictions
            selected_tokens: [B, N, D] - Swin tokens at requested resolution
            pooled_cnn: [B, d_model] - Pooled CNN features
            pooled_swin: [B, d_model] - Pooled Swin features
        """
        # Apply learnable input preprocessing
        x_cnn = x_cnn * self.input_scale_eff + self.input_shift_eff
        x_swin = x_swin * self.input_scale_swin + self.input_shift_swin
        
        # CNN branch
        feats = self.cnn_backbone(x_cnn)  # [B, 1280, H, W]
        
        # Swin branch with hidden states for multi-resolution tokens
        swin_out = self.swin(pixel_values=x_swin, output_hidden_states=True, return_dict=True)
        last_tokens = swin_out.last_hidden_state  # [B, N, D]
        
        # Select tokens at requested resolution
        selected_tokens = last_tokens
        token_stage_l = (token_stage or "last").lower()
        
        if swin_out.hidden_states is not None:
            try:
                import math
                
                def grid_size(t):
                    n = t.shape[1]
                    s = int(math.sqrt(n))
                    return s if s * s == n else None
                
                last_s = grid_size(last_tokens) or 7
                
                # Handle specific resolution requests
                if token_stage_l in ("56", "28", "14", "7"):
                    target = int(token_stage_l)
                    for hs in swin_out.hidden_states:
                        s = grid_size(hs)
                        if s == target:
                            selected_tokens = hs
                            break
                
                # Handle high-resolution request
                elif token_stage_l in ("hr", "stage3", "high", "highres"):
                    candidates = []
                    for hs in swin_out.hidden_states:
                        s = grid_size(hs)
                        if s is not None and s > last_s:
                            candidates.append((s, hs))
                    
                    # Prefer 14x14 resolution
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
        
        # Handle special token stage requests
        try:
            if token_stage_l in ("1", "global", "pool"):
                # Global pooled token [B, 1, D]
                selected_tokens = last_tokens.mean(dim=1, keepdim=True)
            
            elif token_stage_l in ("56", "28", "14", "7"):
                target = int(token_stage_l)
                B, N, D = selected_tokens.shape
                cur_s = int(N ** 0.5) if int(N ** 0.5) ** 2 == N else None
                
                if cur_s is not None and cur_s != target:
                    # Reshape and pool to target resolution
                    toks = selected_tokens.view(B, cur_s, cur_s, D).permute(0, 3, 1, 2).contiguous()
                    toks_p = F.adaptive_avg_pool2d(toks, (target, target))
                    selected_tokens = toks_p.permute(0, 2, 3, 1).contiguous().view(B, target * target, D)
        
        except Exception:
            pass
        
        # Aggregate Swin features for fusion
        pooled_swin = last_tokens.mean(dim=1)
        
        # Process CNN features through fusion (but we need pooled version)
        # For visualization, we'll manually pool CNN features
        B, C, H, W = feats.shape
        pooled_cnn = F.adaptive_avg_pool2d(feats, (1, 1)).view(B, C)
        
        # Run fusion to get final predictions
        fused, cnn_pool_fusion, swin_pool_fusion, alpha, _ = self.fusion(feats, last_tokens, debug=False)
        logits = self.classifier(fused)
        
        return logits, selected_tokens, cnn_pool_fusion, swin_pool_fusion

    def get_last_cnn_layer(self) -> nn.Module:
        """Helper to get the last conv layer for Grad-CAM hooks"""
        return list(self.cnn_backbone.children())[-1]

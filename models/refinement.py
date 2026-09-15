"""
Progressive Segmentation Refinement Level
==========================================

Given X_n = {D_n, L_n, G_n} and the previous logits S_{n-1}, produce a correction:

    U(x,y)   = 1 - max_k softmax(S_{n-1})(x,y,k)      # uncertainty map
    ΔS_n     = RefinementHead(fused_features, S_{n-1}, gated by U)
    S_n      = S_{n-1} + ΔS_n
    M_n      = argmax(S_n)

Locked design decision from the discussion: uncertainty is computed on S_{n-1}
FIRST and used to GATE where the (more expensive) refinement actually runs —
low-uncertainty regions get a cheap pass-through update, high-uncertainty
regions get the full refinement conv stack. This is the primary spatial-adaptive
compute mechanism (as opposed to level-adaptive early stopping, handled
separately at the top-level loop).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationFusion(nn.Module):
    """Fuse D/L/G feature streams into one map at the target (detail) resolution."""
    def __init__(self, d_ch, l_ch, g_ch, out_ch=128):
        super().__init__()
        self.proj_L = nn.Conv2d(l_ch, out_ch, 1)
        self.proj_G = nn.Conv2d(g_ch, out_ch, 1)
        self.proj_D = nn.Conv2d(d_ch, out_ch, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, D, L, G):
        target_size = D.shape[-2:]
        d_feat = self.proj_D(D)
        l_feat = F.interpolate(self.proj_L(L), size=target_size, mode="bilinear", align_corners=False)
        g_feat = F.interpolate(self.proj_G(G), size=target_size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([d_feat, l_feat, g_feat], dim=1))


class RefinementHead(nn.Module):
    """
    Produces ΔS_n (delta logits) from fused features + previous logits,
    spatially gated by uncertainty computed on S_{n-1}.
    """
    def __init__(self, feat_ch, num_classes, uncertainty_thresh=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.uncertainty_thresh = uncertainty_thresh

        # Light path: cheap 1x1 correction, applied everywhere (low-uncertainty regions
        # only ever get this).
        self.light_path = nn.Conv2d(feat_ch + num_classes, num_classes, 1)

        # Heavy path: full conv stack, applied only where gated on (high-uncertainty).
        self.heavy_path = nn.Sequential(
            nn.Conv2d(feat_ch + num_classes, feat_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, num_classes, 1),
        )

    def forward(self, fused_feat, S_prev):
        target_size = fused_feat.shape[-2:]
        if S_prev is None:
            # Level 1: no previous prediction, no uncertainty gating available yet ->
            # run the heavy path everywhere.
            zeros_logits = torch.zeros(
                fused_feat.shape[0], self.num_classes, *target_size, device=fused_feat.device
            )
            x = torch.cat([fused_feat, zeros_logits], dim=1)
            delta = self.heavy_path(x)
            uncertainty_map = torch.ones(
                fused_feat.shape[0], 1, *target_size, device=fused_feat.device
            )
            return delta, uncertainty_map

        S_prev_up = F.interpolate(S_prev, size=target_size, mode="bilinear", align_corners=False)
        probs = F.softmax(S_prev_up, dim=1)
        uncertainty_map = 1.0 - probs.max(dim=1, keepdim=True).values  # [B,1,H,W]

        x = torch.cat([fused_feat, S_prev_up], dim=1)
        light_delta = self.light_path(x)
        heavy_delta = self.heavy_path(x)

        # Soft gate (differentiable at train time): weight toward heavy_delta as
        # uncertainty increases past the threshold. At inference this can be
        # hardened into a binary mask to actually skip heavy_path compute on
        # low-uncertainty regions (see utils/sparse_infer.py).
        gate = torch.sigmoid((uncertainty_map - self.uncertainty_thresh) * 10.0)
        delta = gate * heavy_delta + (1 - gate) * light_delta

        return delta, uncertainty_map


class ProgressiveSegmentationLevel(nn.Module):
    def __init__(self, d_ch, l_ch, g_ch, num_classes, feat_ch=128, uncertainty_thresh=0.3):
        super().__init__()
        self.fusion = SegmentationFusion(d_ch, l_ch, g_ch, out_ch=feat_ch)
        self.refine = RefinementHead(feat_ch, num_classes, uncertainty_thresh)

    def forward(self, X_n, S_prev):
        D, L, G = X_n
        fused = self.fusion(D, L, G)
        delta, uncertainty_map = self.refine(fused, S_prev)

        target_size = fused.shape[-2:]
        if S_prev is None:
            S_n = delta
        else:
            S_prev_up = F.interpolate(S_prev, size=target_size, mode="bilinear", align_corners=False)
            S_n = S_prev_up + delta

        return S_n, uncertainty_map

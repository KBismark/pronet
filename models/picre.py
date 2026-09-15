"""
PICRE: Progressive Incremental Cross-Resolution Encoder
=========================================================

One level of the progressive encoder. Implements, per level n:

    R_n            = NewInfoExtraction(I_n, Upsample(I_{n-1}))       # image-level novelty
    (state proj)   = StateProjection(X_{n-1})                        # predicted "expected" features
    (dD, dL, dG)   = IncrementalFeatureExtraction(R_n, state_proj, X_{n-1})
    D_n = D_{n-1}(upsampled) + dD
    L_n = L_{n-1}(upsampled) + dL
    G_n = G_{n-1}            + dG
    X_n = CrossResolutionFusion(D_n, L_n, G_n)

Design choices locked from the spec:
  - D (detail): high spatial res, heavy new compute
  - L (local):  mid spatial res, moderate new compute
  - G (global): small fixed token grid, persistent, light update, never recomputed from scratch
  - Novelty extraction is two-stage: image-level divergence R_n first (cheap),
    then feature-level incremental extraction conditioned on X_{n-1} (heavier, but
    only ever looks at R_n + previous state, never re-encodes I_n from scratch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dwconv(in_ch, out_ch, k=3, stride=1, dilation=1):
    """Depthwise-separable conv block: cheap, standard building block throughout."""
    pad = (k // 2) * dilation
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, k, stride, pad, dilation=dilation, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ImageNoveltyExtractor(nn.Module):
    """
    Stage A: R_n = R(I_n, Upsample(I_{n-1}))

    Cheap, shallow. Produces a divergence map at I_n's resolution that highlights
    where the new-resolution image actually differs from what upsampling the
    previous input would have predicted. This is what stops the encoder from
    re-processing the whole image at every level.
    """
    def __init__(self, out_ch=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3 * 2, 16, 3, 1, 1, bias=False),   # concat(I_n, upsampled I_{n-1})
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, I_n, I_prev):
        if I_prev is None:
            # Level 1: nothing to compare against -> treat everything as novel.
            I_prev_up = torch.zeros_like(I_n)
        else:
            I_prev_up = F.interpolate(I_prev, size=I_n.shape[-2:], mode="bilinear", align_corners=False)
        diff_input = torch.cat([I_n, I_prev_up], dim=1)
        R_n = self.stem(diff_input)
        return R_n  # [B, out_ch, H_n, W_n]


class StateProjection(nn.Module):
    """
    Predictive projection P(X_{n-1}) -> expected features at the new resolution.
    Kept deliberately cheap (single 1x1/depthwise conv per stream) so it can't
    eat the compute budget it's supposed to save. Budget target: <5% of level FLOPs.
    """
    def __init__(self, d_ch, l_ch, g_ch):
        super().__init__()
        self.proj_D = nn.Conv2d(d_ch, d_ch, 1)
        self.proj_L = nn.Conv2d(l_ch, l_ch, 1)
        self.proj_G = nn.Conv2d(g_ch, g_ch, 1)

    def forward(self, D, L, G, target_D_size, target_L_size):
        D_up = F.interpolate(D, size=target_D_size, mode="bilinear", align_corners=False)
        L_up = F.interpolate(L, size=target_L_size, mode="bilinear", align_corners=False)
        # G is a fixed-size token grid: no spatial resize, it persists as-is.
        return self.proj_D(D_up), self.proj_L(L_up), self.proj_G(G)


class NoveltyGate(nn.Module):
    """
    A_n = sigmoid(gate(H_n, H_hat_n));  N_n = A_n * (H_n - H_hat_n)
    Learned importance map deciding which differences from the prediction
    are actually useful new information vs. noise.
    """
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, H, H_hat):
        A = self.gate(torch.cat([H, H_hat], dim=1))
        return A * (H - H_hat)


class IncrementalFeatureExtractor(nn.Module):
    """
    Stage B: (dD, dL, dG) = E_delta(R_n, X_{n-1})

    Three asymmetric branches per the spec:
      Detail (D): heaviest new compute, operates near full res on R_n.
      Local  (L): moderate compute, mid-res.
      Global (G): lightest — small cross-attention update over a fixed token grid,
                  pulling in a compressed summary of the new evidence rather than
                  reprocessing anything spatially large.
    """
    def __init__(self, novelty_ch, d_ch=64, l_ch=128, g_ch=256, g_tokens=(16, 32)):
        super().__init__()
        self.g_tokens = g_tokens

        # --- Detail branch (heavy) ---
        self.detail_extract = nn.Sequential(
            dwconv(novelty_ch, d_ch, k=3),
            dwconv(d_ch, d_ch, k=3),
        )
        self.detail_gate = NoveltyGate(d_ch)

        # --- Local branch (moderate): operates on downsampled novelty evidence ---
        self.local_extract = nn.Sequential(
            dwconv(novelty_ch, l_ch, k=3, stride=2),
            dwconv(l_ch, l_ch, k=3),
        )
        self.local_gate = NoveltyGate(l_ch)

        # --- Global branch (light): pool novelty to a small token grid, cross-attend ---
        self.global_pool = nn.AdaptiveAvgPool2d(g_tokens)
        self.global_proj = nn.Conv2d(novelty_ch, g_ch, 1)
        self.global_attn = nn.MultiheadAttention(embed_dim=g_ch, num_heads=4, batch_first=True)

    def forward(self, R_n, D_hat, L_hat, G_hat):
        # Detail
        detail_feat = self.detail_extract(R_n)
        detail_feat = F.interpolate(detail_feat, size=D_hat.shape[-2:], mode="bilinear", align_corners=False)
        dD = self.detail_gate(detail_feat, D_hat)

        # Local
        local_feat = self.local_extract(R_n)
        local_feat = F.interpolate(local_feat, size=L_hat.shape[-2:], mode="bilinear", align_corners=False)
        dL = self.local_gate(local_feat, L_hat)

        # Global: pool new evidence into tokens, let existing global memory (query)
        # attend over new evidence (key/value) -> small, cheap, O(tokens^2) not O(HW^2).
        B, C, gh, gw = G_hat.shape
        g_evidence = self.global_proj(self.global_pool(R_n))          # [B, g_ch, gh, gw]
        q = G_hat.flatten(2).transpose(1, 2)                          # [B, gh*gw, g_ch]
        kv = g_evidence.flatten(2).transpose(1, 2)                    # [B, gh*gw, g_ch]
        attn_out, _ = self.global_attn(q, kv, kv)
        dG = attn_out.transpose(1, 2).reshape(B, C, gh, gw)

        return dD, dL, dG


class CrossResolutionFusion(nn.Module):
    """
    Gated cross-attention / cross-conv fusion between D, L, G after the residual
    updates. Each stream gets a chance to be refined by the others, gated so the
    network can ignore irrelevant cross-scale info.

    Implemented with lightweight gated conv fusion (attention between D and L is
    expensive at high res, so we use pooled context injection instead, which is
    the practical version of the Q/K/V spec in the design doc).
    """
    def __init__(self, d_ch, l_ch, g_ch):
        super().__init__()
        self.g_to_l = nn.Conv2d(g_ch, l_ch, 1)
        self.l_to_d = nn.Conv2d(l_ch, d_ch, 1)
        self.d_to_l = nn.Conv2d(d_ch, l_ch, 1)
        self.l_to_g = nn.Conv2d(l_ch, g_ch, 1)

        self.gate_d = nn.Sequential(nn.Conv2d(d_ch * 2, d_ch, 1), nn.Sigmoid())
        self.gate_l = nn.Sequential(nn.Conv2d(l_ch * 3, l_ch, 1), nn.Sigmoid())
        self.gate_g = nn.Sequential(nn.Conv2d(g_ch * 2, g_ch, 1), nn.Sigmoid())

    def forward(self, D, L, G):
        d_size, l_size = D.shape[-2:], L.shape[-2:]

        # Global -> Local
        g_to_l = F.interpolate(self.g_to_l(G), size=l_size, mode="bilinear", align_corners=False)
        # Local -> Detail
        l_to_d = F.interpolate(self.l_to_d(L), size=d_size, mode="bilinear", align_corners=False)
        # Detail -> Local (pooled down)
        d_to_l = F.adaptive_avg_pool2d(self.d_to_l(D), l_size)
        # Local -> Global (pooled down)
        l_to_g = F.adaptive_avg_pool2d(self.l_to_g(L), G.shape[-2:])

        g_d = self.gate_d(torch.cat([D, l_to_d], dim=1))
        D_out = D + g_d * l_to_d

        g_l = self.gate_l(torch.cat([L, g_to_l, d_to_l], dim=1))
        L_out = L + g_l * (g_to_l + d_to_l)

        g_g = self.gate_g(torch.cat([G, l_to_g], dim=1))
        G_out = G + g_g * l_to_g

        return D_out, L_out, G_out


class PICRE(nn.Module):
    """
    Full progressive encoder level.

    forward(I_n, I_prev, X_prev) -> X_n = {D_n, L_n, G_n}

    X_prev is None on level 1 (nothing to reuse yet -> full-ish extraction,
    still routed through the same modules with zeroed previous state).
    """
    def __init__(self, d_ch=64, l_ch=128, g_ch=256, novelty_ch=16, g_tokens=(16, 32)):
        super().__init__()
        self.d_ch, self.l_ch, self.g_ch = d_ch, l_ch, g_ch
        self.g_tokens = g_tokens

        self.image_novelty = ImageNoveltyExtractor(out_ch=novelty_ch)
        self.state_proj = StateProjection(d_ch, l_ch, g_ch)
        self.feature_extractor = IncrementalFeatureExtractor(
            novelty_ch, d_ch=d_ch, l_ch=l_ch, g_ch=g_ch, g_tokens=g_tokens
        )
        self.fusion = CrossResolutionFusion(d_ch, l_ch, g_ch)

    def _init_state(self, B, device, d_size, l_size):
        D0 = torch.zeros(B, self.d_ch, *d_size, device=device)
        L0 = torch.zeros(B, self.l_ch, *l_size, device=device)
        G0 = torch.zeros(B, self.g_ch, *self.g_tokens, device=device)
        return D0, L0, G0

    def forward(self, I_n, I_prev, X_prev):
        B, _, H, W = I_n.shape
        d_size = (H // 4, W // 4)     # detail branch resolution
        l_size = (H // 8, W // 8)     # local branch resolution

        if X_prev is None:
            D_prev, L_prev, G_prev = self._init_state(B, I_n.device, d_size, l_size)
        else:
            D_prev, L_prev, G_prev = X_prev

        # Stage A: image-level novelty
        R_n = self.image_novelty(I_n, I_prev)

        # Predicted ("expected") features at this level's resolution
        D_hat, L_hat, G_hat = self.state_proj(D_prev, L_prev, G_prev, d_size, l_size)

        # Stage B: incremental feature extraction (gated novelty vs. prediction)
        dD, dL, dG = self.feature_extractor(R_n, D_hat, L_hat, G_hat)

        # Residual state update (upsample previous state, add new information)
        D_prev_up = F.interpolate(D_prev, size=d_size, mode="bilinear", align_corners=False) if D_prev.shape[-2:] != d_size else D_prev
        L_prev_up = F.interpolate(L_prev, size=l_size, mode="bilinear", align_corners=False) if L_prev.shape[-2:] != l_size else L_prev
        D_n = D_prev_up + dD
        L_n = L_prev_up + dL
        G_n = G_prev + dG

        # Cross-resolution fusion
        D_n, L_n, G_n = self.fusion(D_n, L_n, G_n)

        return (D_n, L_n, G_n)

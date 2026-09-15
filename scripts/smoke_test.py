#!/usr/bin/env python
"""
Smoke test: no dataset needed. Verifies the model builds, runs a forward pass
at every operating point, backprops, and reports param counts.

IMPORTANT: this was written and syntax-checked but NOT executed in the build
environment (no torch / no network access there) — run this first on your
machine before trusting anything else in the repo.

Usage:
  python scripts/smoke_test.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.progressive_seg import ProgressiveSegNet
from models.losses import ProgressiveSegLoss
from utils.benchmark import count_parameters


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    num_classes = 19  # Cityscapes-shaped
    model = ProgressiveSegNet(
        num_classes=num_classes,
        num_levels=3,
        resolutions=[(128, 256), (256, 512), (512, 1024)],  # small for a fast smoke test
        d_ch=32, l_ch=64, g_ch=64,   # shrunk for speed; use configs/default.yaml values for real runs
        novelty_ch=8,
        g_tokens=(8, 16),
        feat_ch=64,
    ).to(device)

    print(count_parameters(model))

    B = 2
    image = torch.randn(B, 3, 512, 1024, device=device)
    target = torch.randint(0, num_classes, (B, 512, 1024), device=device)

    print("\n--- Forward pass (all levels) ---")
    out = model(image, target_size=target.shape[-2:])
    for i, logit in enumerate(out["logits"], 1):
        print(f"Level {i}: logits shape={tuple(logit.shape)}, "
              f"mask shape={tuple(out['masks'][i-1].shape)}, "
              f"uncertainty mean={out['uncertainty'][i-1].mean().item():.4f}")

    print("\n--- Loss + backward ---")
    criterion = ProgressiveSegLoss(num_levels=3, ignore_index=255)
    loss_dict = criterion(out["logits"], target)
    print(f"Total loss: {loss_dict['total'].item():.4f}")
    loss_dict["total"].backward()
    print("Backward pass OK — gradients computed.")

    print("\n--- Partial forward (max_level=1, latency-sensitive deployment mode) ---")
    out1 = model(image, max_level=1, target_size=target.shape[-2:])
    print(f"Levels actually run: {len(out1['logits'])} (expected 1)")

    print("\n--- Early-stop inference mode ---")
    model.eval()
    with torch.no_grad():
        out_es = model(image, early_stop_uncertainty=0.9, target_size=target.shape[-2:])
    print(f"Levels actually run with early_stop_uncertainty=0.9: {len(out_es['logits'])}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()

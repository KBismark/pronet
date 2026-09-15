"""
ProgressiveSegNet
==================

Top-level model implementing the full loop from the architecture spec:

    I_n, X_{n-1}         --PICRE_n-->        X_n
    X_n, S_{n-1}          --Refinement_n-->  S_n --> M_n

Runs N progressive levels at increasing input resolution, caching X and S
forward. Supports:
  - training: run all N levels, return all M_n / S_n for deep supervision
  - inference: run up to a requested level, or use uncertainty/early-stop
    criteria to decide when to stop (any-level output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .picre import PICRE
from .refinement import ProgressiveSegmentationLevel


class ProgressiveSegNet(nn.Module):
    def __init__(
        self,
        num_classes,
        num_levels=3,
        resolutions=((256, 512), (512, 1024), (1024, 2048)),
        d_ch=64,
        l_ch=128,
        g_ch=256,
        novelty_ch=16,
        g_tokens=(16, 32),
        feat_ch=128,
        uncertainty_thresh=0.3,
        share_refinement_weights=True,
    ):
        super().__init__()
        assert len(resolutions) == num_levels, "must provide one resolution per level"
        self.num_levels = num_levels
        self.resolutions = resolutions
        self.num_classes = num_classes

        # One PICRE module per level (levels are NOT weight-shared: each operates
        # at a different resolution / novelty regime, per the spec).
        self.picre_levels = nn.ModuleList([
            PICRE(d_ch=d_ch, l_ch=l_ch, g_ch=g_ch, novelty_ch=novelty_ch, g_tokens=g_tokens)
            for _ in range(num_levels)
        ])

        # Refinement head: shared across levels by default (parameter-efficient,
        # forces the head to learn "correction" generically rather than
        # level-specific solutions). Set share_refinement_weights=False to give
        # each level its own head instead.
        if share_refinement_weights:
            head = ProgressiveSegmentationLevel(d_ch, l_ch, g_ch, num_classes, feat_ch, uncertainty_thresh)
            self.refinement_levels = nn.ModuleList([head for _ in range(num_levels)])
        else:
            self.refinement_levels = nn.ModuleList([
                ProgressiveSegmentationLevel(d_ch, l_ch, g_ch, num_classes, feat_ch, uncertainty_thresh)
                for _ in range(num_levels)
            ])

    def _make_pyramid(self, image):
        """Resize the input image to each level's target resolution."""
        pyramid = []
        for (h, w) in self.resolutions:
            pyramid.append(F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False))
        return pyramid

    def forward(self, image, max_level=None, early_stop_uncertainty=None, target_size=None):
        """
        image: [B, 3, H, W] full-resolution input (will be resized per-level internally)
        max_level: if set (1-indexed), stop after this many levels (inference-time control)
        early_stop_uncertainty: if set, stop early once mean uncertainty drops below this
                                 value (inference-time adaptive stopping)
        target_size: output size to upsample final logits to (defaults to input size)

        Returns:
          all_logits: list of S_n for every level actually run (for deep supervision in training)
          all_masks:  list of M_n (argmax) for every level actually run
          uncertainty_maps: list of per-level uncertainty maps
        """
        if target_size is None:
            target_size = image.shape[-2:]

        n_run = self.num_levels if max_level is None else min(max_level, self.num_levels)
        pyramid = self._make_pyramid(image)[:n_run]

        X_prev = None
        S_prev = None
        I_prev = None

        all_logits, all_masks, all_uncertainty = [], [], []

        for n in range(n_run):
            I_n = pyramid[n]
            X_n = self.picre_levels[n](I_n, I_prev, X_prev)
            S_n, uncertainty_map = self.refinement_levels[n](X_n, S_prev)

            all_logits.append(S_n)
            all_uncertainty.append(uncertainty_map)
            all_masks.append(S_n.argmax(dim=1))

            # Early-stop check (inference only): if mean uncertainty is already
            # low, further levels aren't likely to earn their compute cost.
            if early_stop_uncertainty is not None and not self.training:
                if uncertainty_map.mean().item() < early_stop_uncertainty:
                    break

            X_prev, S_prev, I_prev = X_n, S_n, I_n

        # Upsample all returned logits to the requested output size for loss/eval.
        all_logits = [F.interpolate(s, size=target_size, mode="bilinear", align_corners=False) for s in all_logits]
        all_masks = [logit.argmax(dim=1) for logit in all_logits]

        return {
            "logits": all_logits,          # list[S_1 ... S_k]
            "masks": all_masks,            # list[M_1 ... M_k]
            "uncertainty": all_uncertainty,
        }

    @torch.no_grad()
    def predict(self, image, max_level=None, early_stop_uncertainty=None):
        """Convenience inference wrapper: returns only the final mask."""
        self.eval()
        out = self.forward(image, max_level=max_level, early_stop_uncertainty=early_stop_uncertainty)
        return out["masks"][-1]

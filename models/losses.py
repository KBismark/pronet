"""
Losses
======

L_total = sum_n lambda_n * CE(M_n, Y)   +   beta * L_progressive

L_progressive = sum_n max(0, L_n - L_{n-1} + margin)

The progressive margin term explicitly rewards each added level for improving
over the previous one, per the spec (Doc 1 §V).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMCrossEntropy(nn.Module):
    """Online hard example mining CE — standard for class-imbalanced urban/scene
    segmentation (Cityscapes-style)."""
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, logits, target):
        pixel_losses = self.criterion(logits, target).view(-1)
        mask = target.view(-1) != self.ignore_index

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            valid_probs = probs.gather(1, target.clamp(min=0).unsqueeze(1)).view(-1)
            valid_probs[~mask] = 1.0  # ignore

            if valid_probs.numel() > self.min_kept:
                kept_thresh, _ = valid_probs[mask].sort()
                threshold = kept_thresh[min(self.min_kept, len(kept_thresh) - 1)]
                threshold = max(threshold.item(), self.thresh)
            else:
                threshold = 1.0

            hard_mask = mask & (valid_probs < threshold)

        if hard_mask.sum() == 0:
            hard_mask = mask

        return pixel_losses[hard_mask].mean()


class ProgressiveSegLoss(nn.Module):
    def __init__(self, num_levels, ignore_index=255, level_weights=None, margin=0.05, progressive_weight=0.1):
        super().__init__()
        self.num_levels = num_levels
        self.margin = margin
        self.progressive_weight = progressive_weight
        self.ce = OHEMCrossEntropy(ignore_index=ignore_index)
        # Default: weight later (higher-res, harder) levels a bit more.
        self.level_weights = level_weights or [1.0] * num_levels

    def forward(self, all_logits, target):
        """
        all_logits: list of S_n (already upsampled to target size), length = levels actually run
        target: [B, H, W] ground truth
        """
        k = len(all_logits)
        per_level_loss = []
        for n in range(k):
            l_n = self.ce(all_logits[n], target)
            per_level_loss.append(l_n)

        weights = self.level_weights[:k]
        deep_supervision_loss = sum(w * l for w, l in zip(weights, per_level_loss)) / sum(weights)

        progressive_loss = torch.tensor(0.0, device=target.device)
        if k > 1:
            terms = []
            for n in range(1, k):
                terms.append(F.relu(per_level_loss[n] - per_level_loss[n - 1] + self.margin))
            progressive_loss = torch.stack(terms).mean()

        total = deep_supervision_loss + self.progressive_weight * progressive_loss

        return {
            "total": total,
            "deep_supervision": deep_supervision_loss,
            "progressive": progressive_loss,
            "per_level": per_level_loss,
        }

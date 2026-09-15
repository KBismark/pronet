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

    def forward(self, image, max_level=None, early_stop_uncertainty=None, target_size=None,
                overlap_refinement=True):
        """
        image: [B, 3, H, W] full-resolution input (will be resized per-level internally)
        max_level: if set (1-indexed), stop after this many levels (inference-time control)
        early_stop_uncertainty: if set, stop early once mean uncertainty drops below this
                                 value (inference-time adaptive stopping)
        target_size: output size to upsample final logits to (defaults to input size)
        overlap_refinement: if True and on CUDA, run Refinement_n concurrently with
                             PICRE_{n+1} on a second stream (see note below). Has no
                             effect on CPU or when early_stop_uncertainty is set, since
                             early stopping needs S_n's uncertainty before deciding
                             whether to launch PICRE_{n+1} at all.

        Returns:
          all_logits: list of S_n for every level actually run (for deep supervision in training)
          all_masks:  list of M_n (argmax) for every level actually run
          uncertainty_maps: list of per-level uncertainty maps

        --- Parallelism note ---
        The true data-dependency graph is:
            X_n depends on X_{n-1}                      (hard, sequential, unavoidable —
                                                           this IS the "no recomputation"
                                                           principle; collapsing it would
                                                           defeat the design)
            R_n depends only on I_n, I_{n-1}             (NOT on X_{n-1} — can be precomputed
                                                           for all levels upfront, in parallel)
            S_n depends on X_n, S_{n-1}                  (sequential across S, but NOT an
                                                           input to PICRE_{n+1})
        Because Refinement_n does not feed PICRE_{n+1} (PICRE only ever consumes X, never
        S), Refinement_n and PICRE_{n+1} have no data dependency on each other once X_n
        exists. We exploit this with a second CUDA stream: as soon as X_n is ready, we
        launch PICRE_{n+1} and Refinement_n concurrently instead of back-to-back. This
        does NOT parallelize the core X_n <- X_{n-1} recurrence (that stays sequential by
        design) — it only removes the artificial serialization between the encoder and
        refinement branches that the naive for-loop was imposing without cause.
        This is disabled automatically when early_stop_uncertainty is set, because in
        that mode we need S_n's uncertainty to decide whether PICRE_{n+1} should run at
        all — starting it speculatively would waste compute on runs we intend to discard.
        """
        if target_size is None:
            target_size = image.shape[-2:]

        n_run = self.num_levels if max_level is None else min(max_level, self.num_levels)

        # I_n depends only on the source image -> precompute the whole pyramid upfront.
        pyramid = self._make_pyramid(image)[:n_run]

        # R_n depends only on (I_n, I_{n-1}), never on X_{n-1} -> precompute all novelty
        # maps upfront too, independent of the sequential encoder loop below.
        novelty_maps = []
        I_prev = None
        for n in range(n_run):
            novelty_maps.append(self.picre_levels[n].image_novelty(pyramid[n], I_prev))
            I_prev = pyramid[n]

        use_overlap = (
            overlap_refinement
            and image.is_cuda
            and early_stop_uncertainty is None
        )
        stream = torch.cuda.Stream() if use_overlap else None

        X_prev = None
        S_prev = None
        all_logits, all_masks, all_uncertainty = [], [], []
        pending_refine = None  # (event, S_n_future placeholder) when overlapping

        main_stream = torch.cuda.current_stream() if use_overlap else None

        for n in range(n_run):
            I_n = pyramid[n]
            R_n = novelty_maps[n]

            # --- PICRE_n: hard sequential dependency on X_prev, cannot be moved earlier ---
            X_n = self.picre_levels[n].forward_with_novelty(I_n, R_n, X_prev)

            if use_overlap and n < n_run - 1:
                # Cross-stream read hazard: the side stream must explicitly wait for
                # the main stream's queued work (X_n, and S_prev if it was itself
                # produced via a pending refinement) before it's safe to read them —
                # issuing the Python call after X_n exists does NOT guarantee the
                # side stream sees a finished tensor without this.
                stream.wait_stream(main_stream)
                with torch.cuda.stream(stream):
                    S_n, uncertainty_map = self.refinement_levels[n](X_n, S_prev)
                # Main stream proceeds immediately to PICRE_{n+1} without waiting on
                # this refinement to finish computing.
                pending_refine = (stream, S_n, uncertainty_map)
                X_prev = X_n
                # S_prev must be updated now, not deferred: refinement_{n+1} needs S_n
                # as a real input next iteration. This is a required data dependency,
                # not bookkeeping — updating the Python tensor handle here is safe
                # (it's just a reference; correctness across the stream boundary is
                # guaranteed by wait_stream above, on the NEXT iteration's launch).
                S_prev = S_n
            else:
                if pending_refine is not None:
                    prev_stream, prev_S_n, prev_uncertainty = pending_refine
                    torch.cuda.current_stream().wait_stream(prev_stream)
                    all_logits.append(prev_S_n)
                    all_uncertainty.append(prev_uncertainty)
                    all_masks.append(prev_S_n.argmax(dim=1))
                    pending_refine = None
                    # S_prev already holds prev_S_n from when it was launched, above.

                S_n, uncertainty_map = self.refinement_levels[n](X_n, S_prev)
                all_logits.append(S_n)
                all_uncertainty.append(uncertainty_map)
                all_masks.append(S_n.argmax(dim=1))

                if early_stop_uncertainty is not None and not self.training:
                    if uncertainty_map.mean().item() < early_stop_uncertainty:
                        X_prev = X_n
                        break

                X_prev, S_prev = X_n, S_n

        # Drain any refinement still pending on the side stream (last level's, if
        # overlap was used throughout).
        if pending_refine is not None:
            prev_stream, prev_S_n, prev_uncertainty = pending_refine
            torch.cuda.current_stream().wait_stream(prev_stream)
            all_logits.append(prev_S_n)
            all_uncertainty.append(prev_uncertainty)
            all_masks.append(prev_S_n.argmax(dim=1))

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
    
    
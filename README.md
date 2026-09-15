# Progressive Dynamic Multi-Resolution Semantic Segmentation

Reference implementation of the progressive/incremental architecture spec:
progressive input resolution → incremental encoder (PICRE) → incremental
segmentation refinement → any-level output.

## Status

Code was written to match the spec exactly and passes `python -m py_compile`
on every file. **It has not been executed** in the build environment — no
`torch` and no network access were available there. Run `scripts/smoke_test.py`
first, on your machine, before anything else. It builds a small version of the
model, runs a forward pass at every level, backprops, and checks `max_level`
and early-stop inference modes. Fix whatever it surfaces before moving to real
data — treat this as a first draft that needs your test pass, not verified
working code.

## Architecture -> code map

| Spec component                                                                   | File                                                                   |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Five principles / overall loop                                                   | `models/progressive_seg.py` (`ProgressiveSegNet.forward`)              |
| PICRE (image novelty → state projection → gated novelty → D/L/G update → fusion) | `models/picre.py`                                                      |
| Progressive segmentation refinement (uncertainty-gated ΔS_n)                     | `models/refinement.py`                                                 |
| Deep supervision + progressive margin loss                                       | `models/losses.py`                                                     |
| Any-level output / early termination                                             | `ProgressiveSegNet.forward(max_level=..., early_stop_uncertainty=...)` |
| Per-level latency benchmarking (M_1...M_N)                                       | `utils/benchmark.py`, `scripts/evaluate.py`                            |

Two design decisions were locked during our discussion, both implemented as follows:

1. **Image-level novelty is a separate stage from feature-level novelty.**
   `ImageNoveltyExtractor` (cheap, shallow) runs first and produces `R_n`;
   `IncrementalFeatureExtractor` (heavier) only ever consumes `R_n` + projected
   previous state, never the raw image directly. See `models/picre.py`.

2. **Uncertainty is computed on S\_{n-1} and used to gate refinement, not just
   report it.** `RefinementHead` computes the uncertainty map first, then
   blends a cheap "light path" and expensive "heavy path" correction via a
   sigmoid gate on that uncertainty. At inference this gate can be hardened
   into a binary mask to actually skip heavy-path compute region-by-region
   (not yet implemented as sparse/masked convolution — currently the heavy
   path still runs densely and is blended; see "Known gaps" below).

## Setup

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
```

## Training

```bash
python scripts/train.py --config configs/default.yaml \
    --dataset cityscapes --data-root /path/to/cityscapes
```

Supported `--dataset` values: `cityscapes`, `ade20k`, `voc`, `coco_stuff`, `camvid`.
Expected directory layouts are documented in each file under `datasets/`.

## Evaluation (per-operating-point table)

```bash
python scripts/evaluate.py --ckpt checkpoints/best.pth --config configs/default.yaml \
    --dataset cityscapes --data-root /path/to/cityscapes
```

Reports mIoU and latency/FPS at every level (M_1...M_N) separately — this
table tests the "one model, many operating points"

## Known gaps / things to verify before trusting results

These were flagged during design and are NOT resolved by writing the code —
they need empirical answers from your first training runs:

1. **Predictive projection cost.** `StateProjection` is currently a single
   1x1 conv per stream (kept deliberately cheap per the <5% FLOPs budget
   agreed earlier). If ablations show the novelty-gating mechanism isn't
   earning its complexity vs. a naive concat+conv cascade, this is the first
   place to simplify further, not add capacity to.
2. **Heavy/light path gating is soft (blended), not sparse at inference.**
   `RefinementHead` computes both paths and blends them — it does not yet
   skip compute on low-uncertainty regions, so the current code will NOT show
   the speed benefit that uncertainty gating is supposed to provide. Real
   speedup requires masked/sparse convolution or region-batched inference
   (e.g. via `torch.masked_select` + reassembly, or a library like
   `torchsparse`/`MinkowskiEngine`-style sparse conv). This is a to-do, not
   yet built — treat current latency numbers as an upper bound on cost, not
   the target.
3. **No baseline comparison code included.** You said to build exactly what
   was spec'd, so PIDNet-S / STDC baseline training scripts are not included
   here. Add them before drawing any accuracy/speed conclusions — a number
   without a baseline run in the same harness isn't comparable.
4. **COCO-Stuff loader assumes pre-converted PNG labels** (the common
   community release format), not raw COCO panoptic JSON. If your COCO-Stuff
   copy is in a different format, `datasets/coco_stuff.py` will need a
   conversion step first.
5. **No FLOPs counter wired in** (`utils/benchmark.py` only measures wall-clock
   latency). Add `fvcore.nn.FlopCountAnalysis` or `thop.profile` if you need
   FLOPs specifically, since the dynamic gating paths make static FLOP
   counting non-trivial (it'll only reflect the dense/unblended cost until
   gap #2 above is resolved).

## Repo layout

```
models/
  picre.py            # progressive encoder level
  refinement.py        # progressive segmentation refinement level
  progressive_seg.py    # top-level model, wires levels together
  losses.py             # deep supervision + progressive margin loss
datasets/
  cityscapes.py, ade20k.py, pascal_voc.py, coco_stuff.py, camvid.py
  base.py               # shared augmentation/loading logic
utils/
  metrics.py            # mIoU / pixel accuracy
  benchmark.py           # latency + param counting
scripts/
  train.py, evaluate.py, smoke_test.py
configs/
  default.yaml
```


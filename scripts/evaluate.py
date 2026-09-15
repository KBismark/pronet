#!/usr/bin/env python

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.progressive_seg import ProgressiveSegNet
from datasets import build_dataset, num_classes_for
from utils.metrics import SegMetric
from utils.benchmark import benchmark_latency, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--skip-latency", action="store_true")
    return p.parse_args()


def build_model(cfg, num_classes):
    m = cfg["model"]
    resolutions = [tuple(r) for r in m["resolutions"]]
    return ProgressiveSegNet(
        num_classes=num_classes,
        num_levels=m["num_levels"],
        resolutions=resolutions,
        d_ch=m["d_ch"], l_ch=m["l_ch"], g_ch=m["g_ch"],
        novelty_ch=m["novelty_ch"], g_tokens=tuple(m["g_tokens"]),
        feat_ch=m["feat_ch"], uncertainty_thresh=m["uncertainty_thresh"],
        share_refinement_weights=m["share_refinement_weights"],
    )


@torch.no_grad()
def evaluate_all_levels(model, loader, num_classes, ignore_index, device):
    model.eval()
    metrics_per_level = [SegMetric(num_classes, ignore_index) for _ in range(model.num_levels)]

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        out = model(images, target_size=targets.shape[-2:])
        for lvl_idx, pred in enumerate(out["masks"]):
            metrics_per_level[lvl_idx].update(pred, targets)

    return [m.compute() for m in metrics_per_level]


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.dataset:
        cfg["train"]["dataset"] = args.dataset
    if args.data_root:
        cfg["train"]["data_root"] = args.data_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = cfg["train"]
    num_classes = num_classes_for(t["dataset"], num_classes=args.num_classes)

    val_set = build_dataset(t["dataset"], t["data_root"], split="val",
                             crop_size=tuple(t["crop_size"]), augment=False,
                             **({"num_classes": args.num_classes} if args.num_classes else {}))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

    model = build_model(cfg, num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    print(f"\nParams: {count_parameters(model)}")

    print("\n=== mIoU per operating point (M_1 ... M_N) ===")
    results = evaluate_all_levels(model, val_loader, num_classes, val_set.ignore_index, device)
    for i, r in enumerate(results, 1):
        print(f"  Level {i} (res={cfg['model']['resolutions'][i-1]}): "
              f"mIoU={r['mIoU']:.4f}  pixel_acc={r['pixel_acc']:.4f}")

    if not args.skip_latency and device == "cuda":
        print("\n=== Latency per operating point ===")
        input_size = (1, 3, *cfg["model"]["resolutions"][-1])
        lat = benchmark_latency(model, input_size=input_size, device=device)
        for level, stats in lat.items():
            print(f"  {level}: {stats['avg_latency_ms']:.2f} ms  ({stats['fps']:.1f} FPS)  res={stats['resolution']}")
    elif device != "cuda":
        print("\n(Skipping latency benchmark: no CUDA device available)")


if __name__ == "__main__":
    main()

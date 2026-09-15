#!/usr/bin/env python
"""
Training entrypoint.

Usage:
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/default.yaml --dataset ade20k --data-root /data/ade20k

Trains with deep supervision across all progressive levels + the progressive
margin loss, and evaluates mIoU at the final level on a held-out val split.
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.progressive_seg import ProgressiveSegNet
from models.losses import ProgressiveSegLoss
from datasets import build_dataset, num_classes_for
from utils.metrics import SegMetric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, default=None, help="override train.dataset")
    p.add_argument("--data-root", type=str, default=None, help="override train.data_root")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num-classes", type=int, default=None, help="override dataset default (e.g. coco_stuff)")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, args):
    if args.dataset:
        cfg["train"]["dataset"] = args.dataset
    if args.data_root:
        cfg["train"]["data_root"] = args.data_root
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.resume:
        cfg["train"]["resume"] = args.resume
    return cfg


def build_model(cfg, num_classes):
    m = cfg["model"]
    resolutions = [tuple(r) for r in m["resolutions"]]
    return ProgressiveSegNet(
        num_classes=num_classes,
        num_levels=m["num_levels"],
        resolutions=resolutions,
        d_ch=m["d_ch"],
        l_ch=m["l_ch"],
        g_ch=m["g_ch"],
        novelty_ch=m["novelty_ch"],
        g_tokens=tuple(m["g_tokens"]),
        feat_ch=m["feat_ch"],
        uncertainty_thresh=m["uncertainty_thresh"],
        share_refinement_weights=m["share_refinement_weights"],
    )


def build_optimizer(cfg, model):
    t = cfg["train"]
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=0.9, weight_decay=t["weight_decay"])


def poly_lr(optimizer, base_lr, epoch, total_epochs, power=0.9, warmup_epochs=5):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        lr = base_lr * (1 - (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)) ** power
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
            out = model(images, target_size=targets.shape[-2:])
            loss_dict = criterion(out["logits"], targets)
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        if cfg["train"]["grad_clip"]:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if i % cfg["train"]["log_interval"] == 0:
            per_level = [f"{l:.3f}" for l in loss_dict["per_level"]]
            print(f"  epoch {epoch} iter {i}/{len(loader)} "
                  f"loss={loss.item():.4f} (deep_sup={loss_dict['deep_supervision'].item():.4f} "
                  f"prog={loss_dict['progressive'].item():.4f}) per_level={per_level}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, num_classes, ignore_index, device):
    model.eval()
    metric = SegMetric(num_classes, ignore_index)
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        out = model(images, target_size=targets.shape[-2:])  # runs all levels
        final_pred = out["masks"][-1]
        metric.update(final_pred, targets)
    return metric.compute()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    t = cfg["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_classes = num_classes_for(t["dataset"], num_classes=args.num_classes)
    print(f"Dataset: {t['dataset']}, num_classes={num_classes}")

    train_set = build_dataset(t["dataset"], t["data_root"], split="train",
                               crop_size=tuple(t["crop_size"]), augment=True,
                               **({"num_classes": args.num_classes} if args.num_classes else {}))
    val_set = build_dataset(t["dataset"], t["data_root"], split="val",
                             crop_size=tuple(t["crop_size"]), augment=False,
                             **({"num_classes": args.num_classes} if args.num_classes else {}))

    train_loader = DataLoader(train_set, batch_size=t["batch_size"], shuffle=True,
                               num_workers=t["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                             num_workers=t["num_workers"], pin_memory=True)

    model = build_model(cfg, num_classes).to(device)
    criterion = ProgressiveSegLoss(
        num_levels=cfg["model"]["num_levels"],
        ignore_index=train_set.ignore_index,
        level_weights=t["level_weights"],
        margin=t["margin"],
        progressive_weight=t["progressive_loss_weight"],
    )
    optimizer = build_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=t["amp"])

    os.makedirs(t["ckpt_dir"], exist_ok=True)
    start_epoch = 0
    best_miou = 0.0

    if t["resume"]:
        ckpt = torch.load(t["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed from {t['resume']} at epoch {start_epoch}")

    for epoch in range(start_epoch, t["epochs"]):
        lr = poly_lr(optimizer, t["lr"], epoch, t["epochs"], t["poly_power"], t["warmup_epochs"])
        print(f"Epoch {epoch} | lr={lr:.6f}")

        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg, epoch)
        print(f"Epoch {epoch} done. avg_loss={avg_loss:.4f}")

        if (epoch + 1) % cfg["eval"]["eval_interval_epochs"] == 0 or epoch == t["epochs"] - 1:
            metrics = evaluate(model, val_loader, num_classes, train_set.ignore_index, device)
            print(f"[Eval] epoch {epoch}: mIoU={metrics['mIoU']:.4f} pixel_acc={metrics['pixel_acc']:.4f}")

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
                "config": cfg,
            }
            torch.save(ckpt, os.path.join(t["ckpt_dir"], "last.pth"))

            if cfg["eval"]["save_best"] and metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                ckpt["best_miou"] = best_miou
                torch.save(ckpt, os.path.join(t["ckpt_dir"], "best.pth"))
                print(f"  New best mIoU: {best_miou:.4f} -> saved best.pth")


if __name__ == "__main__":
    main()

"""Confusion-matrix based mIoU / pixel accuracy, standard for segmentation eval."""

import numpy as np
import torch


class SegMetric:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion.fill(0)

    def update(self, pred, target):
        """pred, target: [B,H,W] tensors of class indices."""
        pred = pred.detach().cpu().numpy().reshape(-1)
        target = target.detach().cpu().numpy().reshape(-1)
        mask = target != self.ignore_index
        pred, target = pred[mask], target[mask]
        valid = (pred >= 0) & (pred < self.num_classes)
        pred, target = pred[valid], target[valid]

        idx = target.astype(np.int64) * self.num_classes + pred.astype(np.int64)
        counts = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.confusion.astype(np.float64)
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = intersection / np.maximum(union, 1e-10)
        valid_classes = cm.sum(axis=1) > 0
        miou = iou[valid_classes].mean() if valid_classes.any() else 0.0
        pixel_acc = intersection.sum() / max(cm.sum(), 1e-10)
        return {"mIoU": float(miou), "pixel_acc": float(pixel_acc), "per_class_iou": iou}

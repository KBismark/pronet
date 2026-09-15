"""
ADE20K loader (scene parsing, 150 classes).

Expected layout (standard ADEChallengeData2016 release):
  root/
    images/{training,validation}/ADE_train_...jpg
    annotations/{training,validation}/ADE_train_...png   (values 0-150, 0=ignore/background)
"""

import glob
import os
from .base import SegmentationDataset

ADE20K_NUM_CLASSES = 150  # labels 1..150 used; 0 reserved for "unlabeled"


class ADE20KDataset(SegmentationDataset):
    num_classes = ADE20K_NUM_CLASSES
    ignore_index = 255
    id_map = None  # remap handled inline (labels are 0..150, subtract 1, 0 -> ignore)

    def __init__(self, root, split="train", crop_size=(1024, 1024), augment=True):
        super().__init__(split=split, crop_size=crop_size, augment=augment)
        subdir = "training" if split == "train" else "validation"

        img_pattern = os.path.join(root, "images", subdir, "*.jpg")
        images = sorted(glob.glob(img_pattern))

        for img_path in images:
            fname = os.path.basename(img_path).replace(".jpg", ".png")
            mask_path = os.path.join(root, "annotations", subdir, fname)
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No ADE20K samples found under {root} for split '{split}'.")

    def _remap_labels(self, mask_np):
        # ADE20K annotation PNGs: 0 = unlabeled, 1..150 = classes.
        # Shift to 0..149 for training classes, 0 -> ignore_index.
        out = mask_np.astype("int64") - 1
        out[mask_np == 0] = self.ignore_index
        return out

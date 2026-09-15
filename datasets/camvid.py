"""
CamVid loader. 11 classes (common SegNet-style split), small dataset -> good
for fast sanity-checking of the training loop before scaling to Cityscapes/ADE20K.

Expected layout (common CamVid release, e.g. from the SegNet/BiSeNet repos):
  root/
    train/*.png            (images)
    train_labels/*.png     (label ids, 0..10, 11=ignore/void)
    val/*.png
    val_labels/*.png
    test/*.png
    test_labels/*.png
"""

import glob
import os
from .base import SegmentationDataset

CAMVID_NUM_CLASSES = 11

CAMVID_CLASSES = [
    "sky", "building", "pole", "road", "pavement", "tree", "signsymbol",
    "fence", "car", "pedestrian", "bicyclist",
]


class CamVidDataset(SegmentationDataset):
    num_classes = CAMVID_NUM_CLASSES
    ignore_index = 255
    id_map = {i: i for i in range(CAMVID_NUM_CLASSES)}  # 11 -> ignore handled below

    def __init__(self, root, split="train", crop_size=(512, 512), augment=True):
        super().__init__(split=split, crop_size=crop_size, augment=augment)
        split = {"train": "train", "val": "val", "test": "test"}.get(split, split)

        img_dir = os.path.join(root, split)
        label_dir = os.path.join(root, f"{split}_labels")
        images = sorted(glob.glob(os.path.join(img_dir, "*.png")))

        for img_path in images:
            fname = os.path.basename(img_path)
            base = fname.replace(".png", "")
            mask_path = os.path.join(label_dir, f"{base}_L.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(label_dir, fname)
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No CamVid samples found under {root} for split '{split}'.")

    def _remap_labels(self, mask_np):
        out = mask_np.copy()
        out[mask_np >= CAMVID_NUM_CLASSES] = self.ignore_index
        return out

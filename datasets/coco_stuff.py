"""
COCO-Stuff loader (172 classes: 80 thing + 91 stuff, using the common 171/172-class
merged label maps distributed as PNGs, rather than re-deriving from raw COCO panoptic
JSON — this matches the widely-used "cocostuff" PNG label release).

Expected layout:
  root/
    images/{train2017,val2017}/<id>.jpg
    annotations/{train2017,val2017}/<id>.png   (values 0..181 style COCO-Stuff ids, 255=ignore)

Use --num-classes to override if using a reduced label set (e.g. 27-class coarse COCO-Stuff).
"""

import glob
import os
from .base import SegmentationDataset

COCO_STUFF_NUM_CLASSES = 171  # standard "COCO-Stuff 171" evaluation setup


class COCOStuffDataset(SegmentationDataset):
    num_classes = COCO_STUFF_NUM_CLASSES
    ignore_index = 255
    id_map = None  # release PNGs are already trainId-ready in the common distribution

    def __init__(self, root, split="train", crop_size=(512, 512), augment=True, num_classes=None):
        super().__init__(split=split, crop_size=crop_size, augment=augment)
        if num_classes is not None:
            self.num_classes = num_classes

        subdir = "train2017" if split == "train" else "val2017"
        img_pattern = os.path.join(root, "images", subdir, "*.jpg")
        images = sorted(glob.glob(img_pattern))

        for img_path in images:
            fname = os.path.basename(img_path).replace(".jpg", ".png")
            mask_path = os.path.join(root, "annotations", subdir, fname)
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No COCO-Stuff samples found under {root} for split '{split}'.")

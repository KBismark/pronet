"""
Pascal VOC 2012 (+ SBD augmentation) loader. 21 classes (20 + background).

Expected layout (standard VOCdevkit release):
  root/
    JPEGImages/<id>.jpg
    SegmentationClass/<id>.png     (palette-indexed, 0=background, 255=ignore/boundary)
    ImageSets/Segmentation/{train,val}.txt
"""

import os
from .base import SegmentationDataset

VOC_NUM_CLASSES = 21

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class PascalVOCDataset(SegmentationDataset):
    num_classes = VOC_NUM_CLASSES
    ignore_index = 255
    id_map = None  # VOC label PNGs already use trainable ids directly (0..20, 255=ignore)

    def __init__(self, root, split="train", crop_size=(512, 512), augment=True):
        super().__init__(split=split, crop_size=crop_size, augment=augment)
        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"VOC split file not found: {split_file}")

        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]

        for img_id in ids:
            img_path = os.path.join(root, "JPEGImages", f"{img_id}.jpg")
            mask_path = os.path.join(root, "SegmentationClass", f"{img_id}.png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No VOC samples found under {root} for split '{split}'.")

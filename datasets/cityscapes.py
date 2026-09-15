"""
Cityscapes loader.

Expected layout:
  root/
    leftImg8bit/{train,val,test}/<city>/<city>_..._leftImg8bit.png
    gtFine/{train,val,test}/<city>/<city>_..._gtFine_labelIds.png

19 training classes, standard Cityscapes labelId -> trainId remap.
"""

import glob
import os
from .base import SegmentationDataset

# Standard Cityscapes labelId -> trainId map (19 classes, rest -> ignore 255)
CITYSCAPES_ID_MAP = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


class CityscapesDataset(SegmentationDataset):
    num_classes = 19
    ignore_index = 255
    id_map = CITYSCAPES_ID_MAP

    def __init__(self, root, split="train", crop_size=(1024, 1024), augment=True):
        super().__init__(split=split, crop_size=crop_size, augment=augment)
        self.root = root

        img_pattern = os.path.join(root, "leftImg8bit", split, "*", "*_leftImg8bit.png")
        images = sorted(glob.glob(img_pattern))

        for img_path in images:
            fname = os.path.basename(img_path)
            city = fname.split("_")[0]
            base = fname.replace("_leftImg8bit.png", "")
            mask_path = os.path.join(root, "gtFine", split, city, f"{base}_gtFine_labelIds.png")
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No Cityscapes samples found under {root} for split '{split}'. "
                f"Check that leftImg8bit/ and gtFine/ are present."
            )

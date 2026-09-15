"""
Common base for all dataset loaders.

Every dataset returns (image_tensor[3,H,W] float, mask_tensor[H,W] long) with
mask values in [0, num_classes-1] or ignore_index (255) for unlabeled pixels.
All resizing to the model's progressive resolutions happens inside the model
(ProgressiveSegNet._make_pyramid); datasets just return full-resolution
(or dataset-native-resolution) image/mask pairs plus light augmentation.
"""

import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SegmentationDataset(Dataset):
    """
    Subclasses must set:
      self.samples: list of (image_path, mask_path) tuples
      self.num_classes: int
      self.ignore_index: int (default 255)
      self.id_map: optional dict remapping raw label ids -> training ids
    """
    num_classes = None
    ignore_index = 255
    id_map = None

    def __init__(self, split="train", crop_size=(1024, 1024), augment=True):
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and split == "train"
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, img_path, mask_path):
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        return image, mask

    def _remap_labels(self, mask_np):
        if self.id_map is None:
            return mask_np
        out = np.full_like(mask_np, self.ignore_index)
        for raw_id, train_id in self.id_map.items():
            out[mask_np == raw_id] = train_id
        return out

    def _augment(self, image, mask):
        # Random scale
        scale = random.uniform(0.75, 1.5)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Random crop (pad if needed)
        ch, cw = self.crop_size
        pad_h, pad_w = max(ch - new_h, 0), max(cw - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=self.ignore_index)

        w2, h2 = image.size
        x = random.randint(0, max(w2 - cw, 0))
        y = random.randint(0, max(h2 - ch, 0))
        image = image.crop((x, y, x + cw, y + ch))
        mask = mask.crop((x, y, x + cw, y + ch))

        # Random horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Color jitter (image only)
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        return image, mask

    def _center_or_resize_eval(self, image, mask):
        ch, cw = self.crop_size
        image = image.resize((cw, ch), Image.BILINEAR)
        mask = mask.resize((cw, ch), Image.NEAREST)
        return image, mask

    def _to_tensors(self, image, mask_np):
        img_t = TF.to_tensor(image)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
        mask_t = torch.from_numpy(mask_np.astype(np.int64))
        return img_t, mask_t

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image, mask = self._load_pair(img_path, mask_path)

        if self.augment:
            image, mask = self._augment(image, mask)
        else:
            image, mask = self._center_or_resize_eval(image, mask)

        mask_np = np.array(mask, dtype=np.int64)
        mask_np = self._remap_labels(mask_np)

        return self._to_tensors(image, mask_np)

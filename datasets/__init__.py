from .cityscapes import CityscapesDataset
from .ade20k import ADE20KDataset
from .pascal_voc import PascalVOCDataset
from .coco_stuff import COCOStuffDataset
from .camvid import CamVidDataset

REGISTRY = {
    "cityscapes": CityscapesDataset,
    "ade20k": ADE20KDataset,
    "voc": PascalVOCDataset,
    "coco_stuff": COCOStuffDataset,
    "camvid": CamVidDataset,
}


def build_dataset(name, root, split="train", crop_size=(1024, 1024), augment=True, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(REGISTRY.keys())}")
    cls = REGISTRY[name]
    return cls(root=root, split=split, crop_size=crop_size, augment=augment, **kwargs)


def num_classes_for(name, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(REGISTRY.keys())}")
    cls = REGISTRY[name]
    if "num_classes" in kwargs and kwargs["num_classes"] is not None:
        return kwargs["num_classes"]
    return cls.num_classes

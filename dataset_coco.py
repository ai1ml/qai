# training/dataset_coco.py
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T


def build_transforms(train: bool):
    # DETR: ToTensor -> Normalize (ImageNet)
    trs = [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(trs)


class CocoDetDataset(Dataset):
    """
    Minimal COCO detection dataset for DETR-style training.
    - Expects COCO bbox in xywh (pixels); converts to xyxy (pixels).
    - Returns image: Tensor [3,H,W], target: dict with 'boxes', 'labels', 'image_id', 'orig_size'
    """
    def __init__(self, images_root: str, ann_file: str, train: bool = True):
        self.images_root = Path(images_root)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = build_transforms(train=train)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_id = self.ids[index]
        rec = self.coco.loadImgs(img_id)[0]
        img_path = self.images_root / rec["file_name"]
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes_xyxy = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(a["category_id"])

        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image = self.transforms(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([rec["height"], rec["width"]], dtype=torch.int64),
        }
        return image, target


def collate_fn(samples: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    images, targets = list(zip(*samples))
    return list(images), list(targets)

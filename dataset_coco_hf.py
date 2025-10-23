# training/dataset_coco_hf.py
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoHFDataset(Dataset):
    """
    COCO → HuggingFace DETR-friendly dataset.
    Returns:
      image: PIL.Image (RGB)
      target: dict with keys 'image_id', 'annotations' (each has bbox[x,y,w,h], category_id, area, iscrowd)
    NOTE:
      - COCO category_id often starts at 1; we remap to 0..N-1 via catid_map if provided.
      - BBoxes are in absolute pixels (x,y,w,h); HF processor will handle conversion/normalization.
    """
    def __init__(self, images_root: str, ann_file: str, catid_map: Dict[int, int] = None):
        self.images_root = Path(images_root)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.catid_map = catid_map or {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict[str, Any]]:
        img_id = self.ids[index]
        rec = self.coco.loadImgs(img_id)[0]
        img_path = self.images_root / rec["file_name"]
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Build HF-style annotations list (xywh in pixels)
        annotations = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            cat = a["category_id"]
            cat = self.catid_map.get(cat, cat)  # remap to 0..N-1 if needed
            annotations.append({
                "bbox": [float(x), float(y), float(w), float(h)],
                "category_id": int(cat),
                "area": float(a.get("area", w*h)),
                "iscrowd": int(a.get("iscrowd", 0)),
            })

        target = {
            "image_id": img_id,
            "annotations": annotations,
        }
        return img, target


def build_category_id_map(coco: COCO, label2id: Dict[str, int]) -> Dict[int, int]:
    """
    Map COCO's category_id (often 1..N) → 0..N-1 in the same order as labels.txt.
    """
    name_to_coco_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    mapping = {}
    for name, new_id in label2id.items():
        coco_id = name_to_coco_id.get(name)
        if coco_id is not None:
            mapping[coco_id] = new_id
    return mapping

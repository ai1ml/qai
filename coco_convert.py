#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supermarket Shelves → COCO Detection converter

- Expects per-image JSON annotations with fields:
    { "size": {"width": W, "height": H},
      "objects": [
         { "classTitle": "Product",
           "points": {"exterior": [[x1,y1],[x2,y2]]} }
      ] }
- Writes:
    out_dir/
      labels.txt
      annotations/
        instances_train.json
        instances_val.json   (or instances_test.json if --holdout test)

Use as a module or CLI.
"""

from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ImgAnnPair:
    image_path: Path
    ann_path: Path


def _read_json(p: Path) -> dict:
    with open(p, "r") as fh:
        return json.load(fh)


def scan_pairs(data_root: Path, images_glob: str = "**/*.jpg") -> List[ImgAnnPair]:
    """Find images and their sibling annotation .json files (e.g., 001.jpg + 001.jpg.json)."""
    pairs: List[ImgAnnPair] = []
    for img in sorted(data_root.rglob(images_glob)):
        ann = img.with_suffix(img.suffix + ".json")  # 001.jpg -> 001.jpg.json
        if ann.exists():
            pairs.append(ImgAnnPair(img, ann))
    return pairs


def parse_supermarket_json(ann_path: Path) -> dict:
    """
    Returns:
      {
        "width": int,
        "height": int,
        "boxes": List[Tuple[class_name, xmin, ymin, xmax, ymax]]
      }
    """
    data = _read_json(ann_path)
    size = data.get("size", {})
    W = size.get("width") or data.get("width")
    H = size.get("height") or data.get("height")
    if W is None or H is None:
        raise ValueError(f"Missing width/height in {ann_path}")

    boxes: List[Tuple[str, float, float, float, float]] = []
    for obj in data.get("objects", []):
        cls = obj.get("classTitle") or obj.get("class") or obj.get("class_name") or "Unknown"
        pts = (obj.get("points") or {}).get("exterior") or []
        if len(pts) >= 2 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in pts[:2]):
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            xmin, ymin = float(min(x1, x2)), float(min(y1, y2))
            xmax, ymax = float(max(x1, x2)), float(max(y1, y2))
            # clamp to image
            xmin = max(0.0, min(xmin, W - 1))
            ymin = max(0.0, min(ymin, H - 1))
            xmax = max(0.0, min(xmax, W - 1))
            ymax = max(0.0, min(ymax, H - 1))
            if xmax > xmin and ymax > ymin:
                boxes.append((cls, xmin, ymin, xmax, ymax))
        # else: ignore malformed
    return {"width": int(W), "height": int(H), "boxes": boxes}


def build_label_list(pairs: List[ImgAnnPair]) -> List[str]:
    """Collect all class names from all annotations; return sorted unique list."""
    classes = set()
    for p in pairs:
        parsed = parse_supermarket_json(p.ann_path)
        for cls, *_ in parsed["boxes"]:
            classes.add(cls)
    labels = sorted(classes)
    return labels


def split_pairs(
    pairs: List[ImgAnnPair],
    train_frac: float,
    holdout: str = "val",
    seed: int = 42,
) -> Tuple[List[ImgAnnPair], List[ImgAnnPair]]:
    """Split into train + (val or test)."""
    assert holdout in ("val", "test"), "--holdout must be 'val' or 'test'"
    rnd = random.Random(seed)
    pairs_shuffled = pairs[:]
    rnd.shuffle(pairs_shuffled)
    n = len(pairs_shuffled)
    n_train = int(n * train_frac)
    train = pairs_shuffled[:n_train]
    hold = pairs_shuffled[n_train:]
    return train, hold


def to_coco_dict(
    pairs: List[ImgAnnPair],
    labels: List[str],
    images_root: Path,
) -> dict:
    """Convert a split into a COCO Detection JSON dict."""
    label_to_id: Dict[str, int] = {name: i + 1 for i, name in enumerate(labels)}  # COCO cat ids start at 1
    images = []
    annotations = []
    categories = [{"id": cid, "name": name, "supercategory": "object"} for name, cid in label_to_id.items()]

    ann_id = 1
    for img_id, pair in enumerate(pairs, start=1):
        parsed = parse_supermarket_json(pair.ann_path)
        W, H = parsed["width"], parsed["height"]

        images.append({
            "id": img_id,
            "file_name": str(pair.image_path.relative_to(images_root).as_posix()),
            "width": W,
            "height": H,
            "date_captured": datetime.utcnow().isoformat()
        })

        for (cls, xmin, ymin, xmax, ymax) in parsed["boxes"]:
            w = xmax - xmin
            h = ymax - ymin
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": label_to_id[cls],
                "iscrowd": 0,
                "bbox": [float(xmin), float(ymin), float(w), float(h)],
                "area": float(w * h),
                "segmentation": []
            })
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def write_text(lines: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def convert_supermarket_to_coco(
    data_root: str,
    out_dir: str,
    train_frac: float = 0.8,
    holdout: str = "val",           # "val" or "test" (choose ONE)
    images_glob: str = "**/*.jpg",
    seed: int = 42,
) -> dict:
    """
    Convert Supermarket Shelves annotations to COCO Detection format.
    Returns dict with written paths.
    """
    data_root_p = Path(data_root).resolve()
    out_dir_p = Path(out_dir).resolve()
    pairs = scan_pairs(data_root_p, images_glob=images_glob)
    if not pairs:
        raise SystemExit(f"No (image, image.json) pairs found under: {data_root_p}")

    labels = build_label_list(pairs)
    labels_path = out_dir_p / "labels.txt"
    write_text(labels, labels_path)

    train_pairs, hold_pairs = split_pairs(pairs, train_frac=train_frac, holdout=holdout, seed=seed)

    train_coco = to_coco_dict(train_pairs, labels, data_root_p)
    hold_coco  = to_coco_dict(hold_pairs,  labels, data_root_p)

    ann_dir = out_dir_p / "annotations"
    train_path = ann_dir / "instances_train.json"
    hold_name  = "instances_val.json" if holdout == "val" else "instances_test.json"
    hold_path  = ann_dir / hold_name

    write_json(train_coco, train_path)
    write_json(hold_coco,  hold_path)

    # Console summary
    print(f"\n✅ Wrote: {labels_path}")
    print(f"✅ Wrote: {train_path}  (images: {len(train_coco['images'])}, annots: {len(train_coco['annotations'])})")
    print(f"✅ Wrote: {hold_path}   (images: {len(hold_coco['images'])}, annots: {len(hold_coco['annotations'])})")

    return {"labels": str(labels_path), "train": str(train_path), "holdout": str(hold_path)}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert Supermarket Shelves dataset to COCO Detection format.")
    p.add_argument("--data-root", required=True, help="Folder with images and per-image .json files")
    p.add_argument("--out-dir",   required=True, help="Folder to write COCO outputs (labels + annotations/...)")
    p.add_argument("--train-frac", type=float, default=0.8, help="Fraction of images for train split")
    p.add_argument("--holdout", choices=("val", "test"), default="val", help="Choose a single holdout split")
    p.add_argument("--images-glob", default="**/*.jpg", help="Glob for images (e.g., **/*.jpg)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split")
    return p


def main():
    args = _build_argparser().parse_args()
    convert_supermarket_to_coco(
        data_root=args.data_root,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        holdout=args.holdout,
        images_glob=args.images_glob,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

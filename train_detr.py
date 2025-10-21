# training/train_detr.py
import os, json, argparse, sys, subprocess, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Light bootstrap for pycocotools if missing (SageMaker base image may not have it) ---
try:
    import pycocotools  # noqa
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocotools==2.0.7"])

from dataset_coco import CocoDetDataset, collate_fn  # local import after pip


# --- Qualcomm AI Hub model ---
from qai_hub_models.models.detr_resnet50.model import DETRResNet50


def parse_args():
    p = argparse.ArgumentParser()
    # Relative paths inside the input channel (S3 prefix)
    p.add_argument("--images_root", default="images", type=str)
    p.add_argument("--train_annotations", default="annotations/instances_train.json", type=str)
    p.add_argument("--val_annotations",   default="annotations/instances_val.json", type=str)
    p.add_argument("--labels_file",       default="labels.txt", type=str)

    # Training HPs
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)

    # SageMaker specifics
    p.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    p.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    return p.parse_args()


def load_labels(labels_path: Path):
    labels = [ln.strip() for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return labels


def reinit_class_head(model: nn.Module, num_classes: int):
    """
    Re-initialize DETR classification head to `num_classes` (plus background).
    Works for common layouts. Raises with a helpful message otherwise.
    """
    # Most DETR variants have an attribute named class_embed (Linear)
    # with shape [hidden_dim, num_classes+1]
    head = None
    attr_chain = ["class_embed", "model.class_embed", "detr.class_embed"]
    for name in attr_chain:
        try:
            head = eval(f"model.{name}")
            if isinstance(head, nn.Linear):
                in_f = head.in_features
                setattr(model, name.split(".")[-1], nn.Linear(in_f, num_classes + 1))
                print(f"[head] Replaced {name} with nn.Linear({in_f}, {num_classes+1})")
                return
        except Exception:
            pass
    raise RuntimeError(
        "Could not locate DETR classification head to adapt. "
        "Ensure this model exposes a 'class_embed' linear layer or provide a helper."
    )


def sum_loss_dict(loss_dict):
    total = 0.0
    log = {}
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            total = total + v
            log[k] = float(v.detach().cpu().item())
    log["total"] = float(total.detach().cpu().item())
    return total, log


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve dataset paths inside the training channel root
    train_root = Path(args.training_dir)
    images_root = train_root / args.images_root
    train_ann = train_root / args.train_annotations
    val_ann   = train_root / args.val_annotations
    labels_fp = train_root / args.labels_file

    assert images_root.exists(), f"images_root not found: {images_root}"
    assert train_ann.exists(), f"train_annotations not found: {train_ann}"
    assert val_ann.exists(),   f"val_annotations not found: {val_ann}"
    assert labels_fp.exists(), f"labels file not found: {labels_fp}"

    labels = load_labels(labels_fp)
    num_classes = len(labels)
    print(f"[data] classes: {num_classes} -> {labels[:10]}{'...' if len(labels)>10 else ''}")

    # Datasets & loaders
    train_ds = CocoDetDataset(str(images_root), str(train_ann), train=True)
    val_ds   = CocoDetDataset(str(images_root), str(val_ann),   train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    # Model
    model = DETRResNet50.from_pretrained()   # COCO pretrained backbone/transformer
    reinit_class_head(model, num_classes)    # adapt head to your classes
    model.to(device)
    model.train()

    # Optimizer (classic DETR settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images, targets)  # DETR returns a dict of losses in train mode
            loss, loss_log = sum_loss_dict(outputs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # quick val (optional): compute loss dict on val
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            count = 0
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                out = model(images, targets)   # loss dict
                l, _ = sum_loss_dict(out)
                val_loss += l.item()
                count += 1
        val_loss = val_loss / max(count, 1)

        dt = time.time() - t0
        print(f"[epoch {epoch:03d}] train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  time={dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            # save best checkpoint
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_dir, "state_dict.pth"))
            with open(os.path.join(args.model_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        "labels": labels,
                        "num_classes": num_classes,
                        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                        "framework": "pytorch",
                        "architecture": "DETRResNet50",
                    },
                    f,
                    indent=2,
                )
            print(f"[save] best -> {args.model_dir}/state_dict.pth  (val_loss={best_val:.4f})")

    print("[done] training complete")


if __name__ == "__main__":
    main()

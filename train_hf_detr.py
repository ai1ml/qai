# training/train_hf_detr.py
import os, sys, json, argparse, subprocess, pkgutil
from pathlib import Path

# --- Ensure deps in container (SageMaker) ---
def ensure(pkg, pip_name=None):
    if pkgutil.find_loader(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg])

ensure("pycocotools", "pycocotools==2.0.7")
ensure("transformers", "transformers==4.44.0")
ensure("accelerate", "accelerate==1.0.1")
ensure("datasets", "datasets==3.0.1")
ensure("evaluate", "evaluate==0.4.2")

from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    TrainingArguments,
    Trainer,
)
from pycocotools.coco import COCO
from dataset_coco_hf import CocoHFDataset, build_category_id_map


def parse_args():
    p = argparse.ArgumentParser()
    # Dataset (relative to training channel root)
    p.add_argument("--images_root", default="images", type=str)
    p.add_argument("--train_annotations", default="annotations/instances_train.json", type=str)
    p.add_argument("--val_annotations",   default="annotations/instances_val.json", type=str)
    p.add_argument("--labels_file",       default="labels.txt", type=str)

    # Model
    p.add_argument("--model_name", default="facebook/detr-resnet-50", type=str)

    # HPs
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--fp16", type=int, default=1)

    # SageMaker dirs
    p.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))

    # Compile-time input size hint (optional; training uses native aug)
    p.add_argument("--compile_height", type=int, default=800)
    p.add_argument("--compile_width",  type=int, default=800)

    return p.parse_args()


def read_labels(path: Path):
    labels = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in enumerate(labels)}
    return labels, id2label, label2id


def main():
    args = parse_args()

    root = Path(args.training_dir)
    images_root = root / args.images_root
    train_ann = root / args.train_annotations
    val_ann   = root / args.val_annotations
    labels_fp = root / args.labels_file

    # Debug prints
    print("[paths]")
    print(" training_dir:", root)
    print(" images_root :", images_root)
    print(" train_ann   :", train_ann)
    print(" val_ann     :", val_ann)
    print(" labels_file :", labels_fp)

    assert images_root.exists(), f"images_root not found: {images_root}"
    assert train_ann.exists(), f"train_annotations not found: {train_ann}"
    assert val_ann.exists(),   f"val_annotations not found: {val_ann}"
    assert labels_fp.exists(), f"labels_file not found: {labels_fp}"

    labels, id2label, label2id = read_labels(labels_fp)
    num_labels = len(labels)
    print(f"[labels] {num_labels} classes -> {labels[:10]}{'...' if len(labels)>10 else ''}")

    # Processor and model
    processor = DetrImageProcessor.from_pretrained(
        args.model_name,
        do_resize=True,                    # DETR uses shortest_edge resize internally
        size={"shortest_edge": 800},       # typical DETR setting; training-time only
        max_size=1333,
    )
    model = DetrForObjectDetection.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,      # allow class head resize
    )

    # Build category_id mapping so that COCO ids → 0..N-1 as per labels.txt
    coco_train = COCO(str(train_ann))
    catid_map = build_category_id_map(coco_train, label2id)

    # Datasets
    train_ds = CocoHFDataset(str(images_root), str(train_ann), catid_map=catid_map)
    val_ds   = CocoHFDataset(str(images_root), str(val_ann),   catid_map=catid_map)

    # Collate via processor (handles box conversion/normalization)
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        return processor(images=list(images), annotations=list(targets), return_tensors="pt")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=bool(args.fp16),
        dataloader_num_workers=4,
        remove_unused_columns=False,   # IMPORTANT for object detection
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # simple criterion
        greater_is_better=False,
    )

    # Minimal metric: use loss from Trainer (HF computes losses internally)
    def compute_metrics(_):
        # Return empty dict to use built-in loss for model selection
        return {}

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("[train] starting...")
    trainer.train()
    print("[train] done.")

    # Save model + processor + label maps for export/compile stage
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)
    with open(save_dir / "labels.txt", "w") as f:
        f.write("\n".join(labels))

    # (Optional) Hint compile-time input spec for later
    with open(save_dir / "compile_input_spec.json", "w") as f:
        json.dump({"images": {"shape": [1, 3, args.compile_height, args.compile_width], "dtype": "float32"}}, f)

    print(f"[save] artifacts at: {save_dir}")


if __name__ == "__main__":
    main()

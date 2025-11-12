import numpy as np
import boto3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

s3 = boto3.client("s3")

# ---- AUTO DISCOVER GROUND-TRUTH CLASSES ----
def get_ground_truth_classes(bucket, prefix):
    classes = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            cls = rel.split("/")[0]
            if cls:
                classes.add(cls)
    return sorted(list(classes))

GT_CLASSES = get_ground_truth_classes(BUCKET, VAL_PREFIX)
print("Auto-detected GT classes:", GT_CLASSES)

# Add "other" for baseline model unknowns
CLASSES = GT_CLASSES + ["other"]

# ---- helper: canonical GT label ----
def canon_true(lbl):
    return lbl.lower()

# ---- helper: map pretrained model labels to 5 flowers ----
def auto_map_pred(raw_label, gt_classes):
    lbl = raw_label.lower()

    # Find best match: if any GT class appears as substring of the raw predicted label
    for cls in gt_classes:
        if cls in lbl:
            return cls

    return "other"   # fallback

# ---- extract GT label from S3 key ----
def true_label_from_key(key):
    return key[len(VAL_PREFIX):].split("/", 1)[0]


# ---- UNIVERSAL VALIDATION FUNCTION ----
def evaluate(predictor, title):
    y_true, y_pred = [], []

    for key in image_keys:
        img_bytes = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        resp      = predictor.predict(img_bytes)

        probs  = resp.get("probabilities", [])
        labels = resp.get("labels", [])

        if not probs or not labels:
            continue

        pred_idx = int(np.argmax(probs))
        raw_pred = labels[pred_idx]

        gt  = canon_true(true_label_from_key(key))
        prd = auto_map_pred(raw_pred, GT_CLASSES)

        y_true.append(gt)
        y_pred.append(prd)

    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=CLASSES)

    print(f"\n=== {title} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0))

    return cm, acc


# ---- RUN FOR BOTH MODELS ----
cm_pre, acc_pre = evaluate(pretrained_predictor, "Pretrained")
cm_ft,  acc_ft  = evaluate(finetuned_predictor, "Fine-tuned")

# ---- SIDE-BY-SIDE CONFUSION MATRICES ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm, acc, title in [
    (axes[0], cm_pre, acc_pre, "Pretrained"),
    (axes[1], cm_ft,  acc_ft,  "Fine-tuned"),
]:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"{title} (Acc = {acc:.2%})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.show()

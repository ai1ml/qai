# ---- Build class list from subfolders (alphabetical = model's index order) ----
def list_classes(bucket, root_prefix):
    cls = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix):
        for o in page.get("Contents", []):
            rel = o["Key"][len(root_prefix):]
            if "/" in rel:
                cls.add(rel.split("/", 1)[0])
    return sorted([c for c in cls if c])

classes = list_classes(BUCKET, VAL_PREFIX)
id_to_class = {i: c for i, c in enumerate(classes)}
print("Classes (index→name):", id_to_class)

# ---- Inference loop: use argmax over 'probabilities' ----
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def true_label_from_key(key):
    return key[len(VAL_PREFIX):].split("/", 1)[0]

y_true, y_pred = [], []
for key in image_keys:
    img_bytes = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    resp = predictor.predict(img_bytes)               # dict with 'probabilities'
    probs = resp.get("probabilities", None)
    if probs is None:
        raise ValueError(f"No 'probabilities' in response: {resp}")
    pred_idx = int(np.argmax(probs))
    pred_label = id_to_class[pred_idx]

    y_true.append(true_label_from_key(key))
    y_pred.append(pred_label)

acc = accuracy_score(y_true, y_pred)
print(f"Validation accuracy on {len(y_true)} images: {acc:.4f}")
print(classification_report(y_true, y_pred, target_names=classes))

# ---- Clean up ----
predictor.delete_endpoint()

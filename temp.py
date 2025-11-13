import random, io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---- config ----
NUM_ROWS = 2   # how many rows of examples you want

# ---- helpers ----
def get_class_from_key(key):
    return key[len(VAL_PREFIX):].split("/", 1)[0]

def get_top1(predictor, img_bytes):
    resp   = predictor.predict(img_bytes)
    labels = resp.get("labels", [])
    probs  = resp.get("probabilities", [])
    if not labels or not probs:
        return "N/A", 0.0
    idx = int(np.argmax(probs))
    return labels[idx], probs[idx]

# ---- gather keys per class ----
image_keys = list_all_images(BUCKET, VAL_PREFIX)

class_to_keys = {}
for k in image_keys:
    cls = get_class_from_key(k)
    class_to_keys.setdefault(cls, []).append(k)

classes = sorted(class_to_keys.keys())
n_classes = len(classes)

# ---- sample keys: NUM_ROWS per class ----
sampled = {}
for cls in classes:
    keys = class_to_keys[cls]
    if len(keys) >= NUM_ROWS:
        sampled[cls] = random.sample(keys, NUM_ROWS)
    else:
        # if not enough images in this class, sample with replacement
        sampled[cls] = random.choices(keys, k=NUM_ROWS)

# ---- create figure ----
fig, axes = plt.subplots(NUM_ROWS, n_classes, figsize=(3*n_classes, 3*NUM_ROWS))
if NUM_ROWS == 1:
    axes = np.expand_dims(axes, 0)   # make it 2D for uniform indexing

for row in range(NUM_ROWS):
    for col, cls in enumerate(classes):
        key = sampled[cls][row]
        img_bytes = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # predictions from both models
        pre_label, pre_prob = get_top1(pretrained_predictor, img_bytes)
        ft_label,  ft_prob  = get_top1(finetuned_predictor, img_bytes)

        ax = axes[row, col]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel(
            f"Pre: {pre_label} ({pre_prob:.2f})\n"
            f"FT : {ft_label} ({ft_prob:.2f})",
            fontsize=8
        )

plt.tight_layout()
plt.show()

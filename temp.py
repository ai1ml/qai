import numpy as np
bad = []

for key in image_keys:
    img = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    r = predictor.predict(img)
    probs = r.get("probabilities", [])
    labels = r.get("labels", [])
    if not probs or not labels:
        continue
    pred = labels[int(np.argmax(probs))]
    gt = true_label_from_key(key)
    if pred.lower() != gt.lower():
        bad.append((key, gt, pred))

print("Total images:", len(image_keys))
print("Misclassified by pretrained model:", len(bad))
print("Sample errors:")
for b in bad[:5]:
    print(f"  {b[0]} — GT: {b[1]} | Pred: {b[2]}")


--------

BUCKET = "qcom-dev-experience"
VAL_PREFIX = "mobilenetv2/datasets/tf_flowers/validation/"
image_keys = list_all_images(BUCKET, VAL_PREFIX)

---------

def true_label_from_key(key):
    return key[len(VAL_PREFIX):].split("/")[0]

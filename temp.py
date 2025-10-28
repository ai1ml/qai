# ---- Deploy temporary endpoint ----
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m5.xlarge")
predictor.serializer = IdentitySerializer("image/jpeg")
predictor.deserializer = JSONDeserializer()

# ---- List ALL validation images ----
import boto3
s3 = boto3.client("s3")

BUCKET = "<YOUR_BUCKET_NAME>"  # e.g., "qcom-dev-experiences"
VAL_PREFIX = "mobilenetv2/datasets/tf_flowers/validation/"

def list_all_images(bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            k = o["Key"]
            if k.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                keys.append(k)
    return keys

image_keys = list_all_images(BUCKET, VAL_PREFIX)
print("Found images:", len(image_keys))

# ---- Run inference & compute accuracy ----
from sklearn.metrics import accuracy_score
import numpy as np

def true_label_from_key(key):
    # folder right under VAL_PREFIX
    return key[len(VAL_PREFIX):].split("/", 1)[0]

def parse_prediction(resp):
    # Handle common JumpStart response shapes
    if isinstance(resp, dict):
        for k in ("predicted_label", "label", "class", "predicted_class"):
            if k in resp and isinstance(resp[k], str):
                return resp[k]
        for k in ("predicted_label", "label", "class_index"):
            if k in resp and isinstance(resp[k], (int, float)):
                return str(int(resp[k]))
        if "result" in resp:
            return parse_prediction(resp["result"])
    if isinstance(resp, list) and resp:
        top = resp[0]
        if isinstance(top, dict) and "label" in top:
            return str(top["label"])
    return str(resp)  # fallback

y_true, y_pred = [], []
for i, key in enumerate(image_keys, 1):
    img_bytes = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    resp = predictor.predict(img_bytes)
    pred = parse_prediction(resp)
    y_true.append(true_label_from_key(key))
    y_pred.append(pred)

acc = accuracy_score(y_true, y_pred)
print(f"Validation accuracy on {len(y_true)} images: {acc:.4f}")

# ---- Clean up ----
predictor.delete_endpoint()

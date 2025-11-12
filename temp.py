import boto3, base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

# ---- CONFIG ----
BUCKET = "your-bucket-name"
PREFIX = "path/to/images/"   # e.g. "mobilenetv2/datasets/tf_flowers/validation/"
MAX_IMAGES = 5               # set None to use ALL images

# ---- configure predictor (once) ----
predictor.serializer   = IdentitySerializer("application/x-image")
predictor.deserializer = JSONDeserializer()

s3 = boto3.client("s3")

def list_image_keys(bucket, prefix):
    """Recursively list all image keys under the given prefix."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                keys.append(k)
    return keys

# ---- get keys (and optionally limit) ----
image_keys = list_image_keys(BUCKET, PREFIX)
if MAX_IMAGES is not None:
    image_keys = image_keys[:MAX_IMAGES]

print(f"Running inference on {len(image_keys)} images\n")

for key in image_keys:
    img_bytes = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    result = predictor.predict(img_bytes)

    labels = result.get("labels", [])
    probs  = result.get("probabilities", [])
    top5   = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:5]

    print(f"Image: s3://{BUCKET}/{key}")
    for label, p in top5:
        print(f"  {label:<25} {p:.4f}")
    print("-" * 60)

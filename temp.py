import boto3, random

# ======= CONFIGURE THESE =======
SRC_BUCKET   = "your-bucket"                  # where all your data lives now
SRC_PREFIX   = "your/data/prefix/"            # e.g. "datasets/tf_flowers/"
DST_BUCKET   = SRC_BUCKET                     # can be same or different bucket
TRAIN_PREFIX = "datasets_split/training/"     # destination prefix for training
VAL_PREFIX   = "datasets_split/validation/"   # destination prefix for validation
TRAIN_RATIO  = 0.95                           # 95% train / 5% val
SEED         = 42
# =================================

s3 = boto3.client("s3")
paginator = s3.get_paginator("list_objects_v2")

# 1) Collect keys
keys = []
for page in paginator.paginate(Bucket=SRC_BUCKET, Prefix=SRC_PREFIX):
    for obj in page.get("Contents", []):
        k = obj["Key"]
        if k.endswith("/") or k == SRC_PREFIX:
            continue
        keys.append(k)

# 2) Split
random.Random(SEED).shuffle(keys)
cut = max(1, int(len(keys) * TRAIN_RATIO))
train_keys, val_keys = keys[:cut], keys[cut:]

# 3) Copy helpers (preserve relative path beneath SRC_PREFIX)
def rel(k): return k[len(SRC_PREFIX):]
def dst_key_train(k): return TRAIN_PREFIX + rel(k)
def dst_key_val(k):   return VAL_PREFIX   + rel(k)

for k in train_keys:
    s3.copy_object(Bucket=DST_BUCKET,
                   CopySource={"Bucket": SRC_BUCKET, "Key": k},
                   Key=dst_key_train(k))
for k in val_keys:
    s3.copy_object(Bucket=DST_BUCKET,
                   CopySource={"Bucket": SRC_BUCKET, "Key": k},
                   Key=dst_key_val(k))

print(f"Total: {len(keys)}  →  train: {len(train_keys)}  val: {len(val_keys)}")
print(f"Training prefix:  s3://{DST_BUCKET}/{TRAIN_PREFIX}")
print(f"Validation prefix: s3://{DST_BUCKET}/{VAL_PREFIX}")

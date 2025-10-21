import sagemaker
from sagemaker.s3 import S3Uploader

bucket = sagemaker.Session().default_bucket()
prefix = "datasets/supermarket_shelves_coco"

# Upload images (directly from raw)
S3Uploader.upload(
    local_path="../data_raw/Supermarket shelves/Supermarket shelves/images",
    desired_s3_uri=f"s3://{bucket}/{prefix}/images"
)

# Upload annotations + labels
S3Uploader.upload(
    local_path="../data_coco/annotations",
    desired_s3_uri=f"s3://{bucket}/{prefix}/annotations"
)
S3Uploader.upload(
    local_path="../data_coco/labels.txt",
    desired_s3_uri=f"s3://{bucket}/{prefix}/labels.txt"
)

print(f"✅ Uploaded to s3://{bucket}/{prefix}/")

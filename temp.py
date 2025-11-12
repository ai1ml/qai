import boto3, os, tarfile, tempfile
from urllib.parse import urlparse

# === EDIT: put your S3 prefix that contains output/model/ ===
S3_DIR = "s3://<bucket>/<prefix>/output/model/"

# Parse S3
p = urlparse(S3_DIR, allow_fragments=False)
bucket, prefix = p.netloc, p.path.lstrip("/")

s3 = boto3.resource("s3")
bkt = s3.Bucket(bucket)

# 1) Download everything under output/model/ to a temp dir
tmpdir = tempfile.mkdtemp()
for obj in bkt.objects.filter(Prefix=prefix):
    if obj.key.endswith("/"): 
        continue
    local_path = os.path.join(tmpdir, os.path.relpath(obj.key, prefix))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    bkt.download_file(obj.key, local_path)

# 2) Create model.tar.gz with files at the top level (as they are now)
tar_local = os.path.join(tmpdir, "model.tar.gz")
with tarfile.open(tar_local, "w:gz") as tar:
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if f == "model.tar.gz": 
                continue
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, tmpdir)  # keep relative paths
            tar.add(full, arcname=arcname)

# 3) Upload tarball next to output/
tar_key = prefix.rsplit("model/", 1)[0] + "model.tar.gz"   # .../output/model.tar.gz
bkt.upload_file(tar_local, tar_key)

model_data = f"s3://{bucket}/{tar_key}"
print("Packaged model at:", model_data)
=========

import sagemaker, boto3
from sagemaker.model import Model
from sagemaker import image_uris

region = boto3.Session().region_name
sess   = sagemaker.Session()
role   = sagemaker.get_execution_role()

# EDIT to your ID/version (3B in your screenshots)
model_id = "meta-textgeneration-llama-3-2-3b-instruct"
model_version = "1.*"

inference_image_uri = image_uris.retrieve(
    region=region, framework=None, model_id=model_id, model_version=model_version
)

sm_model = Model(
    image_uri=inference_image_uri,
    model_data=model_data,                 # <-- the tar we just uploaded
    role=role,
    env={"accept_eula": "true"}            # required for Meta models
)

predictor = sm_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    endpoint_name="llama32-3b-finetuned-endpoint"
)

# quick test
payload = {"inputs":"Summarize: Edge AI reduces latency and cost.", "parameters":{"max_new_tokens":64}}
print(predictor.predict(payload))




import boto3, sagemaker
from sagemaker.model import Model
from sagemaker import image_uris

region = "us-east-1"                      # ✅ correct region
instance_type = "ml.g5.2xlarge"           # ✅ pick a GPU type for Llama 3.2 3B

model_id = "meta-textgeneration-llama-3-2-3b-instruct"
model_version = "1.*"
model_data = "s3://<bucket>/<prefix>/output/model.tar.gz"  # your packaged tarball

image_uri = image_uris.retrieve(
    region=region,
    model_id=model_id,
    model_version=model_version,
    image_scope="inference",              # use model_scope="inference" on newer SDKs
    instance_type=instance_type,          # ✅ required for resolving the right ECR
)

role = sagemaker.get_execution_role()
sm_model = Model(
    image_uri=image_uri,
    model_data=model_data,
    role=role,
    env={"accept_eula": "true"},
)

predictor = sm_model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name="llama32-3b-finetuned-endpoint",
)

-------

from sagemaker.jumpstart.model import JumpStartModel

js_model = JumpStartModel(
    model_id=model_id,
    model_version=model_version,
    model_data=model_data,
    role=role,
    environment={"accept_eula": "true"},
)

predictor = js_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    endpoint_name="llama32-3b-finetuned-endpoint",
)

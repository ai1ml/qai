from sagemaker.analytics import TrainingJobAnalytics

job_name = estimator.latest_training_job.name
tja = TrainingJobAnalytics(training_job_name=job_name)
df = tja.dataframe()

print("Columns:", df.columns.tolist())
print("Unique metric names:", df["metric_name"].unique() if not df.empty else "No metrics found")
print(df.head(10))

----
metrics = df[df["metric_name"].isin([
    "pytorch-ic-train:accuracy", "pytorch-ic-validation:accuracy",
    "pytorch-ic-train:loss", "pytorch-ic-validation:loss"
])]

------

from sagemaker import Model
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

predictor = estimator.deploy(
    instance_type="ml.m5.xlarge",
    initial_instance_count=1,
)
predictor.serializer = IdentitySerializer("image/jpeg")
predictor.deserializer = JSONDeserializer()

# Example prediction for one validation image:
import boto3, random
s3 = boto3.client("s3")

bucket = "<your-bucket>"
prefix = "mobilenetv2/datasets/tf_flowers/validation/daisy/"
resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
key = resp["Contents"][0]["Key"]

img_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
result = predictor.predict(img_bytes)
print(result)

predictor.delete_endpoint()

from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.analytics import TrainingJobAnalytics

# --- Your bucket + new split paths ---
s3_bucket = "<YOUR_BUCKET_NAME>"  # e.g., "qcom-dev-experiences"
training_dataset   = f"s3://{s3_bucket}/mobilenetv2/datasets/tf_flowers/training/"
validation_dataset = f"s3://{s3_bucket}/mobilenetv2/datasets/tf_flowers/validation/"

fine_tune_model_path = f"s3://{s3_bucket}/mobilenetv2/fine-tuned-model/"

# --- Estimator ---
estimator = JumpStartEstimator(
    role=aws_role,
    model_id=model_id,
    model_version=model_version,
    instance_type=training_instance_type,
    instance_count=1,
    max_run=36000,
    output_path=fine_tune_model_path,
    tags=tags,
)

# If supported by the recipe, you can also enable early stopping:
# estimator.set_hyperparameters(early_stopping=True, early_stopping_patience=3)

# --- Train with BOTH channels ---
estimator.fit({
    "training":   training_dataset,    # folder-per-class
    "validation": validation_dataset,  # folder-per-class
})

print("fine tuned model path:", fine_tune_model_path)
print("model path:", estimator.model_data)
print("training job:", estimator.latest_training_job.name)

# --- Pull validation metrics from CloudWatch ---
tja = TrainingJobAnalytics(training_job_name=estimator.latest_training_job.name)
df = tja.dataframe()
print(df.sort_values("timestamp").tail(20))


---

# Reuse the estimator above after .fit(...)
from sagemaker.serializers import IdentitySerializer

# Create a transformer from the trained model
transformer = estimator.transformer(
    instance_type="ml.m5.xlarge",
    instance_count=1,
    assemble_with="Line",
    strategy="SingleRecord",
    output_path=f"s3://{s3_bucket}/mobilenetv2/batch-transform/val-preds/"
)

# For folder-per-class images, many JumpStart image-classification containers accept raw images.
# If needed, you can also generate a JSONL manifest of the validation set and set content_type to application/jsonlines.
transformer.transform(
    data=validation_dataset,            # s3://.../mobilenetv2/datasets/tf_flowers/validation/
    content_type="application/x-image", # typical for direct image inputs
    split_type="None"
)
transformer.wait()

print("Batch output:", transformer.output_path)

# (Optional) Download outputs and compute accuracy locally:
# - Map each prediction back to its ground-truth label (from the folder name).
# - Then compute accuracy with sklearn.
#
# Example sketch (pseudo):
# from sklearn.metrics import accuracy_score
# y_true, y_pred = ..., ...
# print("Validation accuracy:", accuracy_score(y_true, y_pred))

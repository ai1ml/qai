fine_tune_model_path = f"s3://{s3_bucket}/mobilenetv2/fine-tuned-model/"

estimator = JumpStartEstimator(
    role=aws_role,
    model_id=model_id,                    # <- ensure no space/typo
    model_version=model_version,
    instance_type=training_instance_type,
    instance_count=1,
    max_run=36000,
    output_path=fine_tune_model_path,
    tags=tags,
)

# If supported by the model, you can also set early stopping
# estimator.set_hyperparameters(early_stopping=True, early_stopping_patience=3)

# ✅ Add a "validation" channel
estimator.fit({
    "training":   training_dataset,       # s3://... (folder, tar.gz, or manifest)
    "validation": validation_dataset,     # s3://... (same structure as training)
})

print("fine tuned model path:", fine_tune_model_path)
print("model path:", estimator.model_data)
------
from sagemaker.analytics import TrainingJobAnalytics

tja = TrainingJobAnalytics(
    training_job_name=estimator.latest_training_job.name,
    metric_names=["validation:accuracy", "validation:loss", "train:accuracy", "train:loss"]
)
df = tja.dataframe()
print(df.sort_values("timestamp").tail(10))


---
transformer = estimator.transformer(
    instance_type="ml.m5.xlarge",
    instance_count=1,
    assemble_with="Line",
    strategy="SingleRecord"
)

# If you have a manifest (recommended):
#   transformer.transform(data=validation_manifest_s3_uri, content_type="application/jsonlines", split_type="Line")
# If you have a folder/tar of images (depends on the JumpStart inference script expectations):
transformer.transform(
    data=validation_dataset,              # s3://... (manifest or folder)
    content_type="application/x-image",   # adjust if using manifest
    split_type="Line"                     # or None, depending on your format
)
transformer.wait()
print("Batch output:", transformer.output_path)

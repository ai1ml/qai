import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# TensorFlow CPU image is enough (for conversion only)
image_uri = sagemaker.image_uris.retrieve(
    framework="tensorflow",
    region=sess.boto_region_name,
    version="2.12",
    image_scope="training",   # has Python + pip + TF preinstalled
    instance_type="ml.m5.xlarge"
)

# Create processor
processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name=JOB_NAME_PREFIX,
    env={"QAIHUB_API_TOKEN": AI_HUB_API_TOKEN},
)

# Define I/O
inputs = []
outputs = [
    ProcessingOutput(
        source="/opt/ml/processing/output",
        destination=f"s3://{sess.default_bucket()}/qaihub/compile_outputs/"
    )
]

# Run
processor.run(
    code="src/convert_and_compile.py",
    inputs=inputs,
    outputs=outputs,
    arguments=[
        "--s3_model_artifact", S3_MODEL_ARTIFACT,
        "--img_size", str(IMG_SIZE),
        "--target_device_name", TARGET_DEVICE_NAME,
        "--precision", PRECISION,
        "--s3_calib_images", S3_CALIB_IMAGES or "",
    ]
)

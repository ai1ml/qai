import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput

session = sagemaker.Session()
role = sagemaker.get_execution_role()  # if inside SageMaker notebook
region = session.boto_region_name

s3_prefix = "s3://<your-bucket>/datasets/supermarket_shelves_coco/"  # <-- change

print("Region:", region)
print("Role:", role)
print("Dataset S3:", s3_prefix)


----
# HF DLC versions (pick stable)
hf_estimator = HuggingFace(
    entry_point="train_hf_detr.py",
    source_dir="../training",           # contains train_hf_detr.py & dataset_coco_hf.py
    role=role,
    transformers_version="4.44.0",
    pytorch_version="2.3.0",
    py_version="py310",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    hyperparameters={
        "model_name": "facebook/detr-resnet-50",
        "images_root": "images",
        "train_annotations": "annotations/instances_train.json",
        "val_annotations":   "annotations/instances_val.json",
        "labels_file":       "labels.txt",
        "epochs": 25,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "fp16": 1,
        "compile_height": 800,
        "compile_width": 800,
    },
)

hf_estimator.fit({"training": TrainingInput(s3_prefix)})
print("Model artifact:", hf_estimator.model_data)

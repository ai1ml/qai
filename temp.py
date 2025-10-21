import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

session = sagemaker.Session()
region = session.boto_region_name
role = sagemaker.get_execution_role()  # if running inside SageMaker Notebook; else paste your role ARN

# Your dataset S3 prefix built in Phase 1
s3_prefix = "s3://<your-bucket>/datasets/supermarket_shelves_coco/"  # <-- replace

print("Region:", region)
print("Role:", role)
print("Dataset S3:", s3_prefix)



estimator = PyTorch(
    entry_point="train_detr.py",
    source_dir="../training",               # folder containing train_detr.py and dataset_coco.py
    role=role,
    framework_version="1.13",
    py_version="py39",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    hyperparameters={
        "images_root": "images",
        "train_annotations": "annotations/instances_train.json",
        "val_annotations":   "annotations/instances_val.json",  # or instances_test.json if you chose test
        "labels_file":       "labels.txt",
        "epochs": 25,
        "batch_size": 2,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "num_workers": 4,
    },
)

estimator.fit({"training": TrainingInput(s3_prefix)})
print("Model artifact:", estimator.model_data)

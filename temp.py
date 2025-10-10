!pip install qai-hub-models torch torchvision --quiet

import os
import urllib.request
import torch
from qai_hub_models.models.detr_resnet50 import DETRResNet50

# COCO pretrained checkpoint (Facebook official)
checkpoint_url = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
checkpoint_path = "detr-r50-e632da11.pth"



if not os.path.exists(checkpoint_path):
    print(f"⬇️  Downloading pretrained DETR weights from:\n  {checkpoint_url}")
    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
    print(f"✅ Download complete: {checkpoint_path}")
else:
    print(f"✅ Checkpoint already present: {checkpoint_path}")




# Create Qualcomm DETR-ResNet50 model (architecture only)
model = DETRResNet50()
print("✅ Qualcomm DETRResNet50 architecture created.")

print("Loading COCO pretrained weights into DETRResNet50...")

state_dict = torch.load(checkpoint_path, map_location="cpu")

# Allow partial key mismatches (common between Facebook and Qualcomm versions)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"✅ Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")




# COCO has 91 classes. If you have different number of classes:
YOUR_NUM_CLASSES = 5  # <-- change this
if hasattr(model, "class_embed"):
    in_features = model.class_embed.in_features
    model.class_embed = torch.nn.Linear(in_features, YOUR_NUM_CLASSES)
    print(f"🔁 Replaced classifier head for {YOUR_NUM_CLASSES} classes.")
else:
    print("⚠️ Model does not have a 'class_embed' head — check model definition.")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

print(f"✅ Model ready for fine-tuning on {device}.")


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# DataLoader + training steps here



torch.save(model.state_dict(), "checkpoint_finetuned.pt")



inputs = {
    "train": "s3://sagemaker-sample-files/datasets/image/object-detection/coco/train",
    "val":   "s3://sagemaker-sample-files/datasets/image/object-detection/coco/val"
}


aws s3 ls s3://sagemaker-sample-files/datasets/image/object-detection/coco/train/

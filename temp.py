from sagemaker import jumpstart

# List model IDs by task (for example: image classification)
model_ids = jumpstart.retrieve_jumpstart_model_ids(model_filter="task == 'image-classification'")
print("Available JumpStart model IDs:")
for mid in model_ids[:10]:
    print(mid)

----
from sagemaker import jumpstart

model_id = "tensorflow-ic-mobilenet-v3-small-imagenet1k-classification"

# Retrieve metadata
metadata = jumpstart.retrieve_jumpstart_model_metadata(model_id=model_id)
print("Model Name:", metadata["model_name"])
print("Model ID:", metadata["model_id"])
print("Model Version:", metadata["version"])
print("Framework:", metadata["framework"])
print("Task:", metadata["task"])
-----

import boto3
sm = boto3.client("sagemaker")

# List your registered models
resp = sm.list_models(MaxResults=10)
for m in resp["Models"]:
    print("Model name:", m["ModelName"])

------

from sagemaker import jumpstart
import pandas as pd

# 1️⃣ List ALL JumpStart model IDs available in your region
all_model_ids = jumpstart.retrieve_jumpstart_model_ids()
print(f"Total models available: {len(all_model_ids)}")

# Convert to DataFrame for easy viewing
df_all = pd.DataFrame(all_model_ids, columns=["model_id"])
df_all.head(10)


-----
filtered_ic = jumpstart.retrieve_jumpstart_model_ids(model_filter="task == 'image-classification'")
print(f"Total image-classification models: {len(filtered_ic)}")
pd.DataFrame(filtered_ic, columns=["model_id"]).head(10)


----

filtered_tf = jumpstart.retrieve_jumpstart_model_ids(model_filter="framework == 'tensorflow'")
print(f"TensorFlow models: {len(filtered_tf)}")

filtered_pt = jumpstart.retrieve_jumpstart_model_ids(model_filter="framework == 'pytorch'")
print(f"PyTorch models: {len(filtered_pt)}")

-----

keyword = "mobilenet"
filtered_keyword = [mid for mid in all_model_ids if keyword.lower() in mid.lower()]
print(f"Models containing '{keyword}': {len(filtered_keyword)}")
pd.DataFrame(filtered_keyword, columns=["model_id"])

-------
model_id = filtered_ic[0]  # pick the first image-classification model
metadata = jumpstart.retrieve_jumpstart_model_metadata(model_id=model_id)

# Display useful fields
print("Model Name:", metadata["model_name"])
print("Model ID:", metadata["model_id"])
print("Version:", metadata["version"])
print("Framework:", metadata["framework"])
print("Task:", metadata["task"])
print("Training supported:", metadata["supports_training"])

--------

filter_expr = "task == 'image-classification' and framework == 'tensorflow'"
filtered_tf_ic = jumpstart.retrieve_jumpstart_model_ids(model_filter=filter_expr)
pd.DataFrame(filtered_tf_ic, columns=["model_id"]).head(10)

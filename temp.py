from sagemaker.huggingface import HuggingFaceModel

hf_model = HuggingFaceModel(
    model_data=hf_estimator.model_data,        # S3 path from training
    role=role,
    transformers_version="4.49.0",
    pytorch_version="2.3.0",
    py_version="py310",
    source_dir="serving",                      # contains inference.py
    entry_point="inference.py",
    env={
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",          # optional; avoids hub calls at startup
    },
)

predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
)

-----

from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

predictor.serializer   = IdentitySerializer("application/x-image")
predictor.deserializer = JSONDeserializer()

with open("local_test.jpg","rb") as f:
    result = predictor.predict(f.read())

print(result)  # {"detections":[{"score":..,"label":"class_name","box_xyxy":[x1,y1,x2,y2]}, ...]}

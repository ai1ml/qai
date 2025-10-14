import numpy as np
import json
from sagemaker import get_execution_role
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
from IPython.display import HTML, display

# ==== Configuration ====
aws_role = get_execution_role()
endpoint_instance_type = "ml.g4dn.xlarge"

# Choose a JumpStart image classification model (example: MobileNetV2)
model_id = "pytorch-image-classification-mobilenet-v2"

# ==== Deploy the model ====
model = JumpStartModel(
    model_id=model_id,
    model_version="*",
    role=aws_role
)

# This spins up a real-time endpoint and returns a Predictor object
predictor = model.deploy(
    initial_instance_count=1,
    instance_type=endpoint_instance_type,
)

# ==== Configure serializer/deserializer ====
# Use IdentitySerializer for raw bytes and JSONDeserializer for response
predictor.serializer = IdentitySerializer(content_type="application/x-image")
predictor.deserializer = JSONDeserializer()
predictor.accept = "application/json;verbose"  # get probabilities + labels

# ==== Define inference helpers ====
def query_endpoint(image_bytes: bytes):
    """Send image bytes to endpoint and get predictions"""
    return predictor.predict(image_bytes)

def parse_prediction(model_predictions: dict):
    """Extract label, probabilities, and labels list"""
    predicted_label = model_predictions.get("predicted_label")
    probabilities = model_predictions.get("probabilities", [])
    labels = model_predictions.get("labels", [])
    return predicted_label, probabilities, labels

def top_k(probabilities, labels, k=5):
    """Return top-k class names and probabilities"""
    idx = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:k]
    return [labels[i] for i in idx], [probabilities[i] for i in idx]

# ==== Run inference on a local image ====
image_path = "/Users/sachmittal/Desktop/100080576_f52e8ee070_n.jpg"

with open(image_path, "rb") as f:
    payload = f.read()

# Query endpoint
result = query_endpoint(payload)
print("Raw output:", result)

# Parse and interpret
predicted_label, probabilities, labels = parse_prediction(result)
top5_labels, top5_probs = top_k(probabilities, labels, 5)

# Display result
display(HTML(
    f"<img src='{image_path}' width='250px'/>"
    f"<figcaption><b>Predicted:</b> {predicted_label}</figcaption>"
    f"<figcaption><b>Top-5:</b> {top5_labels}</figcaption>"
))
print("Predicted label:", predicted_label)
print("Top-5 labels:", top5_labels)
print("Top-5 probabilities:", top5_probs)

# ==== Optional: clean up endpoint ====
# predictor.delete_endpoint()

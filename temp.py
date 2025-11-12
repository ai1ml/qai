import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
from IPython.display import HTML, display

# --- Configure predictor ---
predictor.serializer = IdentitySerializer("application/x-image")
predictor.deserializer = JSONDeserializer()

# --- Path to your local image ---
image_path = "/Users/sachinmittal/Desktop/test_flower.jpg"

# --- Read image bytes ---
with open(image_path, "rb") as f:
    img_bytes = f.read()

# --- Run inference ---
result = predictor.predict(img_bytes)

# --- Parse response ---
labels = result.get("labels", [])
probs = result.get("probabilities", [])

# --- Get Top-5 ---
top5 = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:5]

# --- Display result ---
img_b64 = base64.b64encode(img_bytes).decode("utf-8")
display(HTML(f'<img src="data:image/jpeg;base64,{img_b64}" width="250"/><br>'))
print("Top-5 Predictions:\n")
for label, p in top5:
    print(f"{label:<30} {p:.4f}")

import numpy as np
from PIL import Image
import qai_hub as hub

# 1) Device (same one you used for compile/profile)
device = hub.Device("Samsung Galaxy S24 (Family)")

# 2) Load & preprocess image → float32, NCHW, (1,3,224,224)
img_path = r"C:\path\to\your\image.jpg"  # <-- change
img = Image.open(img_path).convert("RGB").resize((224, 224))

x = np.asarray(img, dtype=np.float32) / 255.0   # HWC [0,1]
x = np.transpose(x, (2, 0, 1))                  # CHW
x = np.expand_dims(x, 0)                        # NCHW -> (1,3,224,224)

inputs = {"images": x}   # <-- MUST match the key used at compile

# 3) Run inference on the compiled model
compiled_model = compile_job.get_target_model()   # from your completed compile job
infer_job = hub.submit_inference_job(
    model=compiled_model,
    device=device,
    inputs=inputs
)
infer_job.wait()

# 4) Fetch and parse outputs
out = infer_job.download_output_data()

# out is a dict; grab the first output tensor regardless of its name
out_key = next(iter(out["outputs"]))
logits = np.array(out["outputs"][out_key])       # shape (1, num_classes)

# Optional: softmax + Top-5
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
top5_idx = probs[0].argsort()[-5:][::-1]
print("Top-5 indices:", top5_idx)
print("Top-5 probs:", probs[0][top5_idx])

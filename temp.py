# Pretty inference display for AI Hub compiled model (NCHW input)
import base64, io, numpy as np
from PIL import Image
from IPython.display import HTML, display
import qai_hub as hub

# ----- CONFIG -----
img_path = r"C:\path\to\your\image.jpg"       # change this
device    = hub.Device("Samsung Galaxy S24 (Family)")
compiled_model = compile_job.get_target_model()  # from your completed compile

# Optional class names (replace with your dataset labels)
class_names = ["class_0","class_1","class_2","class_3","class_4"]

# ----- Load & preprocess (float32, NCHW 1x3x224x224) -----
img = Image.open(img_path).convert("RGB")
img_disp = img.resize((250, 250))  # for display only
img = img.resize((224, 224))

x = np.asarray(img, dtype=np.float32) / 255.0  # HWC
x = np.transpose(x, (2, 0, 1))                 # CHW
x = np.expand_dims(x, 0)                       # NCHW
inputs = {"images": x}                          # must match compile input_specs key

# ----- Run remote inference -----
infer_job = hub.submit_inference_job(model=compiled_model, device=device, inputs=inputs)
infer_job.wait()
out = infer_job.download_output_data()

# ----- Extract logits -> softmax -> Top-5 -----
out_key = next(iter(out["outputs"]))                 # e.g., "output_0" or "logits"
logits = np.array(out["outputs"][out_key])           # shape (1, num_classes)
probs  = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
probs  = probs[0]
top5   = probs.argsort()[-5:][::-1]
top1   = top5[0]

# Map to labels if provided
def label(i): 
    return class_names[i] if 0 <= i < len(class_names) else f"class_{i}"

top5_labels = [label(i) for i in top5]
top5_probs  = [float(probs[i]) for i in top5]
pred_label  = label(top1)

# ----- Build pretty HTML -----
buf = io.BytesIO()
img_disp.save(buf, format="JPEG")
img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

html = f"""
<div style="display:flex;gap:16px;align-items:flex-start;">
  <img src="data:image/jpeg;base64,{img_b64}" style="width:250px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1);" />
  <div style="font-family:sans-serif;">
    <div style="font-size:18px;margin-bottom:6px;"><b>Predicted:</b> {pred_label}</div>
    <div style="font-size:14px;margin-bottom:6px;"><b>Top-5 labels:</b> {top5_labels}</div>
    <div style="font-size:14px;"><b>Top-5 probabilities:</b> {[round(p,4) for p in top5_probs]}</div>
  </div>
</div>
"""
display(HTML(html))

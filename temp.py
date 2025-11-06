# AFTER your compile completes
schema = compile_job.shapes  # or compiled_model.shapes if you already have it
print("Expected input names:", list(schema["inputs"].keys()))
print("Expected input dtypes/shapes:", schema["inputs"])

---

from transformers import AutoTokenizer
import numpy as np
import qai_hub as hub

seq_len = 32
tok = AutoTokenizer.from_pretrained("./fine-tuned-llama-1b", use_fast=True)
enc = tok("Explain edge AI in one sentence.",
          return_tensors="np", padding="max_length", truncation=True, max_length=seq_len)

# Keep ONLY the key your schema shows (likely 'input_ids')
input_ids_np = enc["input_ids"].astype(np.int32)

inputs = {"input_ids": [input_ids_np]}   # <-- list of arrays, not a single array!
print("Inputs provided:", {k: np.array(v).shape for k, v in inputs.items()})

---

device = hub.Device("Samsung Galaxy S24")

compiled_model = compile_job.get_target_model()  # or hub.get_model(model_id=...)

infer_job = hub.submit_inference_job(
    model=compiled_model,
    device=device,
    inputs=inputs,
)

status = infer_job.wait()
print("Inference status:", status.code, "-", status.message)
if status.code != "SUCCEEDED":
    raise RuntimeError(f"Inference failed: {status.message}")

out = infer_job.download_output_data()
print("Output keys:", list(out.keys()))

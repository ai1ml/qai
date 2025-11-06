# === Profile compiled Llama model on a real Snapdragon device ===
import qai_hub as hub

# must match your compile target
target_device = hub.Device("Samsung Galaxy S24")
model_id = "llama-1b-compiled"   # any label you like

# compile_job must already exist from submit_compile_job(...)
profile_job = hub.submit_profile_job(
    model=compile_job.get_target_model(),  # compiled target from your earlier step
    device=target_device,
    name=model_id,
)
profile_job.wait()

profile = profile_job.download_profile()
lat_ms = profile["execution_summary"]["estimated_inference_time"] / 1000.0
peak_mb = profile["execution_summary"]["estimated_inference_peak_memory"] / (1024 * 1024)
print(f"[Profile] Estimated latency: {lat_ms:.2f} ms | Peak memory: {peak_mb:.1f} MB")

-----
# === Inference on the compiled Llama model ===
import numpy as np
from transformers import AutoTokenizer
import qai_hub as hub

# config (keep seq_len consistent with what you traced)
model_path = "./fine-tuned-llama-1b"      # your local tokenizer/model dir
prompt = "Explain edge AI in one sentence."
seq_len = 32
target_device = hub.Device("Samsung Galaxy S24")

# tokenize → int32 → pad/truncate to seq_len
tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
enc = tok(prompt, return_tensors="np", padding="max_length",
          truncation=True, max_length=seq_len)
input_ids_np = enc["input_ids"].astype(np.int32)
inputs = {"input_ids": input_ids_np}   # key must match the one you used during tracing

# submit inference
infer_job = hub.submit_inference_job(
    model=compile_job.get_target_model(),
    device=target_device,
    inputs=inputs,
)
infer_job.wait()
out = infer_job.download_output_data()

# inspect keys and get logits (often 'output_0')
print("Output keys:", list(out.keys()))
logits = np.array(out.get("output_0", list(out.values())[0]))

# greedy one-step: pick argmax at the last time step
if logits.ndim == 3:          # (batch, seq_len, vocab)
    next_token_id = int(logits[0, -1].argmax())
elif logits.ndim == 2:        # (seq_len, vocab)
    next_token_id = int(logits[-1].argmax())
else:
    raise ValueError(f"Unexpected logits shape: {logits.shape}")

generated_ids = enc["input_ids"][0].tolist() + [next_token_id]
text = tok.decode(generated_ids, skip_special_tokens=True)
print("\n[Greedy 1-step completion]")
print(text)

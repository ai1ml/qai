import numpy as np
from transformers import AutoTokenizer
import qai_hub as hub

# === CONFIGURATION ===
model_path = "./fine-tuned-llama-1b"     # local tokenizer/model directory
prompt = "Explain edge AI in one sentence."
seq_len = 32
device = "Samsung Galaxy S24"            # match your compile target

# === 1. LOAD TOKENIZER ===
tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
enc = tok(prompt, return_tensors="np", padding="max_length",
          truncation=True, max_length=seq_len)

# === 2. BUILD INPUTS (must match compile schema) ===
input_ids_np = enc["input_ids"].astype(np.int32)   # use int64 only if schema says so
attn_mask_np = enc["attention_mask"].astype(np.int32)

inputs = {
    "input_ids": [input_ids_np],
    "attention_mask": [attn_mask_np],
}

# === 3. GET COMPILED MODEL OBJECT ===
compiled_model = compile_job.get_target_model()

# === 4. RUN INFERENCE ===
infer_job = hub.submit_inference_job(
    model=compiled_model,
    device=device,
    inputs=inputs,
)
infer_job.wait()

# === 5. DOWNLOAD OUTPUT ===
out = infer_job.download_output_data()
print("Output keys:", list(out.keys()))

# Extract logits (usually 'output_0')
logits = np.array(out.get("output_0", list(out.values())[0]))

# === 6. SIMPLE GREEDY DECODE ===
if logits.ndim == 3:        # (batch, seq_len, vocab)
    next_token_id = int(logits[0, -1].argmax())
elif logits.ndim == 2:      # (seq_len, vocab)
    next_token_id = int(logits[-1].argmax())
else:
    raise ValueError(f"Unexpected logits shape: {logits.shape}")

generated_ids = enc["input_ids"][0].tolist() + [next_token_id]
text = tok.decode(generated_ids, skip_special_tokens=True)

print("\n[Greedy 1-step completion]")
print(text)

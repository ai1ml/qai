# === Inference on compiled Llama model (schema: input_ids -> (1, 32) int32) ===
import numpy as np
from transformers import AutoTokenizer
import qai_hub as hub

# config
model_path = "./fine-tuned-llama"        # your local tokenizer dir
prompt = "Explain edge AI in one sentence."
seq_len = 32
device = hub.Device("Samsung Galaxy S24")

# 1) tokenize -> np.int32 -> (1,32)
tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
enc = tok(prompt, return_tensors="np", padding="max_length",
          truncation=True, max_length=seq_len)

input_ids_np = enc["input_ids"].astype(np.int32)          # (1, 32) int32
inputs = {"input_ids": [input_ids_np]}                     # list of numpy arrays

# 2) submit inference against the compiled target model
compiled_model = compile_job.get_target_model()            # compile_job already used for profiling
infer_job = hub.submit_inference_job(
    model=compiled_model,
    device=device,
    inputs=inputs,
)
infer_job.wait()

# 3) fetch and inspect outputs
out = infer_job.download_output_data()
print("Output keys:", list(out.keys()))
logits = np.array(out.get("output_0", list(out.values())[0]))

# 4) simple greedy 1-step decode
if logits.ndim == 3:          # (batch, seq_len, vocab)
    next_token_id = int(logits[0, -1].argmax())
elif logits.ndim == 2:        # (seq_len, vocab)
    next_token_id = int(logits[-1].argmax())
else:
    raise ValueError(f"Unexpected logits shape: {logits.shape}")

generated_ids = enc["input_ids"][0].tolist() + [next_token_id]
print(tok.decode(generated_ids, skip_special_tokens=True))

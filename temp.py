from transformers import AutoTokenizer
import numpy as np
import qai_hub as hub

# === CONFIG ===
SEQ_LEN = 32
PROMPT = "Explain edge AI in one sentence."
DEVICE_NAME = "Samsung Galaxy S24"

# === LOAD TOKENIZER AND BUILD INPUT ===
tok = AutoTokenizer.from_pretrained("./fine-tuned-llama-1b", use_fast=True)
enc = tok(
    PROMPT,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=SEQ_LEN
)

# ✅ FIX: Wrap input array inside a list, and cast to int32
inputs = {"input_ids": [enc["input_ids"].astype(np.int32)]}

# === RUN INFERENCE ===
device = hub.Device(DEVICE_NAME)
compiled_model = compile_job.get_target_model()

infer_job = hub.submit_inference_job(
    model=compiled_model,
    device=device,
    inputs=inputs
)

infer_job.wait()
out = infer_job.download_output_data()

# === CHECK OUTPUT ===
if out is None:
    print("❌ Inference failed. Check job URL:", infer_job.url)
else:
    print("✅ Output keys:", list(out.keys()))
    logits = np.array(out.get("output_0", list(out.values())[0]))
    print("Logits shape:", logits.shape)

    # === SIMPLE GREEDY DECODE ===
    next_token_id = int(logits[0, -1].argmax() if logits.ndim == 3 else logits[-1].argmax())
    generated = enc["input_ids"][0].tolist() + [next_token_id]
    print("\n🧠 Generated:", tok.decode(generated, skip_special_tokens=True))

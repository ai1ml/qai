# --- [1] Prepare inputs for inference ---
from transformers import AutoTokenizer
import numpy as np

model_path = "./fine-tuned-llama/"
prompt = "Explain the benefits of on-device AI."
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(model_path)
enc = tokenizer(prompt, truncation=True, padding=False, max_length=MAX_LEN)

input_ids_np = np.asarray(enc["input_ids"], dtype=np.int32)
attention_mask_np = np.asarray(enc["attention_mask"], dtype=np.int32)

inputs = {
    "input_ids": [input_ids_np[0]],
    "attention_mask": [attention_mask_np[0]],
}

# --- [2] Run inference (your previous inference cell) ---
infer_job = hub.submit_inference_job(
    model=compiled_model.get_target_model(),
    inputs=inputs,
)
infer_job.await_completion()
print(infer_job.outputs)

----
compiled_model = hub.get_model("<your_compiled_model_id>")
print(compiled_model.input_schema)

compiled_model.describe()

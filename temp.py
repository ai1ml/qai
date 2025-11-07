# If you still have the job object
print("job attrs:", [a for a in dir(compile_job) if "shape" in a])
print("submit-time shapes:", getattr(compile_job, "shapes", None))
print("target shapes:", getattr(compile_job, "target_shapes", None))  # after job succeeds

-----

schema = getattr(compile_job, "target_shapes", None) or getattr(compile_job, "shapes", None)
assert schema, "No schema on job; is the job finished?"

# e.g., {'input_ids': ((1, 32), 'int32')}
inp_name, (inp_shape, inp_dtype) = list(schema.items())[0]
print("Expecting:", inp_name, inp_shape, inp_dtype)

# Tokenize and conform to required dtype/shape
from transformers import AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained("./fine-tuned-llama-1b", use_fast=True)
enc = tok("Explain edge AI in one sentence.", return_tensors="np",
          padding="max_length", truncation=True, max_length=inp_shape[1])
arr = enc["input_ids"].astype(np.int32 if inp_dtype == "int32" else np.float32)
arr = arr.reshape(inp_shape)  # ensure (1, seq_len)

inputs = {inp_name: arr}

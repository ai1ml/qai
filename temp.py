import qai_hub as hub
import json

def get_io(compiled_or_job):
    # Try both: compiled model or job
    obj = compiled_or_job
    if hasattr(compiled_or_job, "get_target_model"):
        try:
            obj = compiled_or_job.get_target_model()
        except Exception:
            pass

    # Try common attribute names
    for attr in ["shapes", "shape", "io_schema", "schema", "input_specs"]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            try:
                # Some are callables
                val = val() if callable(val) else val
            except Exception:
                pass
            return attr, val

    # Last resort: inspect __dict__
    return "__dict__", getattr(obj, "__dict__", {})

# --- use it ---
attr_name, schema = get_io(compile_job)  # or compiled_model if you have it
print("IO attribute:", attr_name)
print(json.dumps(schema, indent=2, default=str))


----


# Example: {"inputs": {"input_ids": {"dtype":"int32","shape":[1,32]}}, ...}
inp_section = schema.get("inputs") or schema  # sometimes it's already the dict
if isinstance(inp_section, dict) and all(isinstance(v, dict) for v in inp_section.values()):
    input_name = list(inp_section.keys())[0]
    input_dtype = inp_section[input_name].get("dtype", "int32")
------

# Example: {"inputs": [{"name":"input_ids","dtype":"int32","shape":[1,32]}], ...}
inp_list = schema.get("inputs") or schema.get("input_tensors") or schema
if isinstance(inp_list, list) and len(inp_list) > 0 and isinstance(inp_list[0], dict):
    entry = inp_list[0]
    input_name = entry.get("name", "input_ids")
    input_dtype = entry.get("dtype", "int32")

----

import numpy as np
from transformers import AutoTokenizer

seq_len = 32
tok = AutoTokenizer.from_pretrained("./fine-tuned-llama-1b", use_fast=True)
enc = tok("Explain edge AI in one sentence.", return_tensors="np",
          padding="max_length", truncation=True, max_length=seq_len)

# Cast to the dtype from schema
np_dtype = {"int32": np.int32, "int64": np.int64,
            "float32": np.float32, "float16": np.float16}.get(input_dtype, np.int32)

arr = enc["input_ids"].astype(np_dtype)

# IMPORTANT: value must be a LIST of arrays
inputs = {input_name: [arr]}
print("Prepared inputs:", input_name, [a.shape for a in inputs[input_name]])


----


device = hub.Device("Samsung Galaxy S24")
compiled_model = compile_job.get_target_model()  # or however you retrieve it

job = hub.submit_inference_job(model=compiled_model, device=device, inputs=inputs)
status = job.wait()
print("Status:", status.code, "-", status.message)
if status.code != "SUCCEEDED":
    raise RuntimeError(status.message)

out = job.download_output_data()
print("Output keys:", list(out.keys()))

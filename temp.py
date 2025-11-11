# === Use Bitext Customer Support dataset (Q&A) ===
from datasets import load_dataset

USE_TAGS_IN_CONTEXT = True  # set False to omit category/intent

bitext = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

def to_js_format(ex):
    instr = (ex.get("instruction") or "").strip()
    resp  = (ex.get("response") or "").strip()
    if USE_TAGS_IN_CONTEXT:
        cat = (ex.get("category") or "").strip()
        intent = (ex.get("intent") or "").strip()
        ctx = f"category:{cat}; intent:{intent}".strip(" ;")
    else:
        ctx = ""
    return {"instruction": instr, "context": ctx, "response": resp}

bitext = bitext.map(to_js_format, remove_columns=[c for c in bitext.column_names
                                                  if c not in ["instruction","context","response"]])

# 90/10 split
train_and_test_dataset = bitext.train_test_split(test_size=0.1, seed=42)

# Save JSONL for JumpStart (one record per line)
train_and_test_dataset["train"].to_json("train.jsonl", lines=True)
train_and_test_dataset["test"].to_json("val.jsonl", lines=True)  # optional validation channel

print(train_and_test_dataset)
print(train_and_test_dataset["train"][0])

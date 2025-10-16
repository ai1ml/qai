# --- Extract outputs and apply stable softmax if needed ---
out_key = next(iter(out["outputs"]))
raw = np.array(out["outputs"][out_key])  # shape (1, num_classes)

def looks_like_probs(a):
    return a.ndim == 2 and np.all(a >= -1e-6) and np.all(a <= 1+1e-6) \
           and np.allclose(a.sum(axis=1), 1.0, atol=1e-3)

if looks_like_probs(raw):
    probs = raw[0]  # already probabilities
else:
    logits = raw
    logits = logits - logits.max(axis=1, keepdims=True)  # stabilize
    probs = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))[0]

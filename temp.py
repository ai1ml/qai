def true_label_from_key(key):
    return key[len(VAL_PREFIX):].split("/", 1)[0]

# NEW: normalize GT folder names to 5 flowers
def canon_true(lbl):
    l = lbl.lower()
    return {"roses": "rose", "sunflowers": "sunflower", "tulips": "tulip"}.get(l, l)

# NEW: map any model label → one of 5 flowers or "other"
def map_pred_to_6(label: str) -> str:
    l = (label or "").lower()
    if "daisy" in l:      return "daisy"
    if "dandelion" in l:  return "dandelion"
    if "sunflower" in l:  return "sunflower"
    if "tulip" in l:      return "tulip"
    if "rose" in l:       return "rose"
    return "other"  # anything else = failure

-------

    resp = predictor.predict(img_bytes)  # dict with 'probabilities'
    probs  = resp.get("probabilities", None)
    labels = resp.get("labels", [])

    if probs is None or not labels:
        raise ValueError(f"No 'probabilities' or 'labels' in response: {resp}")

    pred_idx       = int(np.argmax(probs))
    raw_pred_label = labels[pred_idx]

    gt = canon_true(true_label_from_key(key))      # one of 5 flowers
    y_true.append(gt)
    y_pred.append(map_pred_to_6(raw_pred_label))   # 5 flowers or "other"

-----------

classes = ["daisy", "dandelion", "rose", "sunflower", "tulip", "other"]

acc = accuracy_score(y_true, y_pred)
print(f"Validation accuracy on {len(y_true)} images: {acc:.4f}")
print(classification_report(y_true, y_pred,
                            labels=classes,
                            target_names=classes,
                            zero_division=0))

------

cm = confusion_matrix(y_true, y_pred, labels=classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)

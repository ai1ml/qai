import operator, functools
import torch.nn as nn

def _get_by_path(root, path):
    return functools.reduce(getattr, [root] + path.split("."))

def _set_by_path(root, path, value):
    parent_path, attr = path.rsplit(".", 1)
    parent = _get_by_path(root, parent_path) if parent_path else root
    setattr(parent, attr, value)

def reinit_class_head(model: nn.Module, num_classes: int, coco_classes_plus_bg: int = 92):
    """
    Find the classification Linear by SHAPE (out_features == COCO+bg default, usually 92)
    and replace it with nn.Linear(in_features, num_classes+1).
    Works even if the attribute isn't named 'class_embed'.
    """
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            candidates.append((name, m.in_features, m.out_features))

    # Prefer layers that look like COCO heads (out=92), else fall back to the largest out layer
    head = None
    for name, inf, outf in candidates:
        if outf == coco_classes_plus_bg:
            head = (name, inf, outf); break
    if head is None and candidates:
        head = max(candidates, key=lambda t: t[2])  # largest out_features as fallback

    if head is None:
        raise RuntimeError("No Linear layers found in model to swap as class head. "
                           "Enable debug print of model.named_modules().")

    name, in_f, out_f = head
    new_head = nn.Linear(in_f, num_classes + 1)
    _set_by_path(model, name, new_head)
    print(f"[head] Replaced '{name}' Linear({in_f}->{out_f}) with Linear({in_f}->{num_classes+1})")

# Runs inside the HF SageMaker inference container
import io, json, torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

def model_fn(model_dir):
    model = DetrForObjectDetection.from_pretrained(model_dir).eval()
    processor = DetrImageProcessor.from_pretrained(model_dir)
    return {"model": model, "processor": processor}

def input_fn(request_body, request_content_type):
    if request_content_type in ("application/x-image", "image/jpeg", "image/png"):
        return Image.open(io.BytesIO(request_body)).convert("RGB")
    # allow JSON with S3 path or base64 if you want later
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(img, ctx):
    model = ctx["model"]; processor = ctx["processor"]
    enc = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**enc)

    # Post-process to get boxes/scores/labels in original image size
    target_sizes = torch.tensor([img.size[::-1]])  # (H,W)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]

    # Map label ids to strings if present in config
    id2label = getattr(model.config, "id2label", None)
    to_name = (lambda i: id2label.get(int(i), str(int(i)))) if id2label else (lambda i: int(i))

    dets = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        dets.append({
            "score": float(score),
            "label": to_name(label),
            "box_xyxy": [float(x) for x in box.tolist()],
        })
    return {"detections": dets}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"

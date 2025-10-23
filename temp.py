def collate_fn(batch):
    images, targets = list(zip(*batch))  # lists of PIL.Image and dicts
    enc = processor(
        images=list(images),
        annotations=list(targets),
        return_tensors="pt",
    )
    # Return only tensors + the labels list (what DETR expects)
    return {
        "pixel_values": enc["pixel_values"],  # Float tensor [B,3,H,W]
        "labels": enc["labels"],              # List[Dict], stays as Python objects
    }

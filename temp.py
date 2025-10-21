from tools.coco_convert import convert_supermarket_to_coco

paths = convert_supermarket_to_coco(
    data_root="data_raw",          # <-- change to your raw dataset root
    out_dir="data_coco",           # outputs go here
    train_frac=0.8,
    holdout="val",                 # or "test" (choose ONE)
    images_glob="**/*.jpg",
    seed=42,
)
paths

------
import json, pandas as pd, numpy as np
from pathlib import Path

def coco_to_df(coco_dict):
    rows = []
    img_map = {im["id"]: im for im in coco_dict["images"]}
    cat_map = {c["id"]: c["name"] for c in coco_dict["categories"]}
    for a in coco_dict["annotations"]:
        im = img_map[a["image_id"]]
        x, y, w, h = a["bbox"]
        rows.append({
            "file_name": im["file_name"],
            "width": im["width"], "height": im["height"],
            "class": cat_map[a["category_id"]],
            "xmin": x, "ymin": y, "xmax": x+w, "ymax": y+h,
            "bbox_w": w, "bbox_h": h, "area": w*h
        })
    return pd.DataFrame(rows)

train_coco = json.load(open(paths["train"], "r"))
hold_coco  = json.load(open(paths["holdout"], "r"))

for title, coco in [("TRAIN", train_coco), ("HOLDOUT", hold_coco)]:
    df = coco_to_df(coco)
    print(f"\n=== {title} ===")
    print("images:", len(coco["images"]), "  annots:", len(coco["annotations"]))
    if df.empty:
        continue
    print("class counts:", df["class"].value_counts().to_dict())
    df["img_area"] = df["width"] * df["height"]
    cov = df.groupby("file_name").apply(lambda d: d["area"].sum()/d["img_area"].iloc[0]).mean()*100
    print(f"mean bbox coverage per image: {cov:.2f}%")
    print("bbox_w mean/min/max:", round(df["bbox_w"].mean(),1), df["bbox_w"].min(), df["bbox_w"].max())
    print("bbox_h mean/min/max:", round(df["bbox_h"].mean(),1), df["bbox_h"].min(), df["bbox_h"].max())
    ar = (df["bbox_w"]/df["bbox_h"]).replace([np.inf,-np.inf], np.nan).dropna()
    print("aspect ratio mean:", round(ar.mean(),3))
    oob = ((df["xmin"]<0)|(df["ymin"]<0)|(df["xmax"]>df["width"])|(df["ymax"]>df["height"])).sum()
    zero = ((df["bbox_w"]<=0)|(df["bbox_h"]<=0)).sum()
    print("sanity — out_of_bounds:", int(oob), " zero_or_negative:", int(zero))


--------



# Only if/when you're ready to train in SageMaker
import sagemaker
from sagemaker.s3 import S3Uploader

S3_BUCKET = "<your-bucket>"
S3_PREFIX = "supermarket_coco/v1"   # version your dataset

s3_uri = S3Uploader.upload("data_coco", f"s3://{S3_BUCKET}/{S3_PREFIX}/")
print("Uploaded to:", s3_uri)

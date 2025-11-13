import pandas as pd

rows = CLASSES  # e.g., ['daisy','dandelion','rose','sunflower','tulip','other']

data = {}
for metric in ["precision", "recall", "f1-score"]:
    data[f"Pre_{metric}"] = [rep_pre[r][metric] for r in rows]
    data[f"FT_{metric}"]  = [rep_ft[r][metric]  for r in rows]

metrics_df = pd.DataFrame(data, index=rows)

print("Per-class metrics (Pretrained vs Fine-tuned):")
display(metrics_df)

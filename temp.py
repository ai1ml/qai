from pathlib import Path

# Start from your project root guess. Adjust the .. as needed from your notebook location.
ROOT = Path("../data_raw")

print("PWD:", Path.cwd())
print("ROOT:", ROOT.resolve())

# Find the deepest folder that contains both 'images' and 'annotations'
candidates = []
for p in ROOT.rglob("*"):
    if p.is_dir():
        imgs = (p / "images").is_dir()
        anns = (p / "annotations").is_dir()
        if imgs and anns:
            candidates.append(p)

print("Candidates with images/ + annotations/:")
for c in candidates:
    print(" -", c.resolve())

# If you see exactly one printed, that's your true DATA_ROOT.

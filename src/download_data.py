import os
from datasets import load_dataset

ds = load_dataset("wellCh4n/tomato-leaf-disease-image")
OUT = "data/raw"

# short folder names for the 10 classes
NAME_MAP = {
    "A healthy tomato leaf": "Healthy",
    "A tomato leaf with Leaf Mold": "Leaf_Mold",
    "A tomato leaf with Target Spot": "Target_Spot",
    "A tomato leaf with Late Blight": "Late_Blight",
    "A tomato leaf with Early Blight": "Early_Blight",
    "A tomato leaf with Bacterial Spot": "Bacterial_Spot",
    "A tomato leaf with Septoria Leaf Spot": "Septoria_Leaf_Spot",
    "A tomato leaf with Tomato Mosaic Virus": "ToMV",
    "A tomato leaf with Tomato Yellow Leaf Curl Virus": "TYLCV",
    "A tomato leaf with Spider Mites Two-spotted Spider Mite": "Spider_Mites",
}

count = 0
for split in ds.keys():              # merge train + validation; we re-split 70/15/15 later
    data = ds[split]
    names = data.features["label"].names
    for i, row in enumerate(data):
        cls = NAME_MAP[names[row["label"]]]
        folder = os.path.join(OUT, cls)
        os.makedirs(folder, exist_ok=True)
        row["image"].convert("RGB").save(os.path.join(folder, f"{split}_{i}.jpg"), quality=95)
        count += 1
        if count % 2000 == 0:
            print("Saved", count)

print("Done. Saved", count, "images")

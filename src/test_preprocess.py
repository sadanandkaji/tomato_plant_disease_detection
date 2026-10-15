import os, cv2, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import crop_roi, enhance_contrast
from dataset import TomatoDataset, make_loaders

root = "data/split/train"
cls = "Late_Blight"
f = random.choice(os.listdir(f"{root}/{cls}"))
orig = cv2.cvtColor(cv2.imread(f"{root}/{cls}/{f}"), cv2.COLOR_BGR2RGB)
cropped = crop_roi(orig)
resized = cv2.resize(cropped, (224, 224))
clahe = enhance_contrast(resized)
ds = TomatoDataset(root, train=True)
aug = ds.augment(orig)

titles = ["Original", "ROI crop", "Resize 224", "CLAHE contrast", "Augmented"]
imgs = [orig, cropped, resized, clahe, aug]
fig, ax = plt.subplots(1, 5, figsize=(16, 4))
for a, t, im in zip(ax, titles, imgs):
    a.imshow(im); a.set_title(t); a.axis("off")
plt.tight_layout()
plt.savefig("preprocess_demo.png", dpi=150)
print("Saved preprocess_demo.png")

# check one batch and sampler balance
train_dl, val_dl, test_dl, classes = make_loaders(bs=32)
x, y = next(iter(train_dl))
print("Classes:", classes)
print("Batch tensor:", x.shape, x.dtype, "labels:", y.shape)

counts = np.zeros(len(classes), int)
for i, idx in enumerate(list(train_dl.sampler)[:3000]):
    counts[train_dl.dataset.samples[idx][1]] += 1
print("Samples drawn per class (3000 draws):")
for c, n in zip(classes, counts):
    print(f"  {c}: {n}")

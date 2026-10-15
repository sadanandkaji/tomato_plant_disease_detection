import os, sys, torch, numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
sys.path.insert(0, os.path.dirname(__file__))
from model import TomatoCNN
from dataset import make_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("models/best.pth", map_location=device)
classes = ckpt["classes"]
model = TomatoCNN(len(classes)).to(device)
model.load_state_dict(ckpt["state"]); model.eval()

_, _, test_dl, _ = make_loaders(bs=32)
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in tqdm(test_dl):
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()

print("Test accuracy:", round(accuracy_score(y_true, y_pred), 4))
print(classification_report(y_true, y_pred, target_names=classes, digits=3))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")

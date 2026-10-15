import os, sys, time, torch, torch.nn as nn
from torch.utils.data import Subset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from model import TomatoCNN
from dataset import make_loaders

QUICK_TEST = True     # True: tiny CPU run to check for bugs. False: full training
EPOCHS = 30
LR, BS = 1e-3, 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
train_dl, val_dl, test_dl, classes = make_loaders(bs=BS)

if QUICK_TEST:
    EPOCHS = 2
    train_dl = DataLoader(Subset(train_dl.dataset, range(0, 12446, 40)), BS, shuffle=True, num_workers=0)
    val_dl = DataLoader(Subset(val_dl.dataset, range(0, 2669, 20)), BS, num_workers=0)

model = TomatoCNN(len(classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

def run(dl, train):
    model.train(train)
    tot_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x); loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot_loss / n, correct / n

os.makedirs("models", exist_ok=True)
best, patience, bad = 0, 6, 0
hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(EPOCHS):
    t0 = time.time()
    tl, ta = run(train_dl, True)
    vl, va = run(val_dl, False)
    sched.step(vl)
    for k, v in zip(hist, (tl, ta, vl, va)): hist[k].append(v)
    print(f"Ep {epoch+1}/{EPOCHS} | train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f} | {time.time()-t0:.0f}s")
    if va > best:
        best, bad = va, 0
        torch.save({"state": model.state_dict(), "classes": classes}, "models/best.pth")
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping"); break

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)
print("Best val acc:", best, "| saved models/best.pth and training_curves.png")

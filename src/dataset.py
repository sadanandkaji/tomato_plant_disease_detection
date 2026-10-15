import os, cv2, random
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from preprocess import preprocess

class TomatoDataset(Dataset):
    def __init__(self, root, train=False):
        self.train = train
        self.classes = sorted(os.listdir(root))
        self.samples = [(os.path.join(root, c, f), i)
                        for i, c in enumerate(self.classes)
                        for f in os.listdir(os.path.join(root, c))]

    def __len__(self):
        return len(self.samples)

    def augment(self, img):
        if random.random() < 0.5: img = cv2.flip(img, 1)
        if random.random() < 0.5: img = cv2.flip(img, 0)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-25, 25),
                                    random.uniform(0.85, 1.2))   # rotate + zoom
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return np.clip(img * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.train:
            img = self.augment(img)
        return preprocess(img), label

def make_loaders(root="data/split", bs=32):
    tr, va, te = [TomatoDataset(f"{root}/{s}", s == "train") for s in ("train", "val", "test")]
    labels = [l for _, l in tr.samples]
    counts = np.bincount(labels)
    weights = [1.0 / counts[l] for l in labels]          # oversample rare classes
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return (DataLoader(tr, bs, sampler=sampler, num_workers=0),
            DataLoader(va, bs, num_workers=0),
            DataLoader(te, bs, num_workers=0), tr.classes)

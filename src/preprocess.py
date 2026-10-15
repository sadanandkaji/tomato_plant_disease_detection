import cv2, numpy as np, torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def crop_roi(img):
    """Crop to the leaf region using an HSV mask (green/yellow/brown)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (10, 30, 30), (90, 255, 255))
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:              # mask failed, keep full image
        return img
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]

def enhance_contrast(img):
    """CLAHE on the L channel to handle uneven lighting."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def preprocess(img_rgb, size=224):
    img = crop_roi(img_rgb)
    img = cv2.resize(img, (size, size))
    img = enhance_contrast(img)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1).float()   # CHW tensor

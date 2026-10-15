import os, sys, json, io
import cv2, numpy as np, torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import TomatoCNN
from preprocess import preprocess

ckpt = torch.load(os.path.join(ROOT, "models", "best.pth"), map_location="cpu")
CLASSES = ckpt["classes"]
model = TomatoCNN(len(CLASSES)); model.load_state_dict(ckpt["state"]); model.eval()
with open(os.path.join(ROOT, "backend", "remediation.json"), encoding="utf-8-sig") as f:
    REMEDY = json.load(f)

app = FastAPI(title="Tomato Leaf Disease API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health():
    return {"status": "ok", "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a valid image file")
    x = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    top = torch.topk(probs, 3)
    top3 = [{"class": CLASSES[i], "confidence": round(p, 4)} for p, i in zip(top.values.tolist(), top.indices.tolist())]
    best = top3[0]
    return {
        "prediction": best["class"],
        "confidence": best["confidence"],
        "low_confidence": best["confidence"] < 0.6,
        "top3": top3,
        "info": REMEDY.get(best["class"], {}),
    }

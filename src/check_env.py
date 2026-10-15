import sys, torch, torchvision, cv2, numpy, sklearn, fastapi

print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("OpenCV      :", cv2.__version__)
print("NumPy       :", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("FastAPI     :", fastapi.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

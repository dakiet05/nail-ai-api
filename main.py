import io
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from torchvision import models, transforms
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Friendly routes ----------
@app.get("/")
def root():
    return {
        "status": "ok",
        "see": ["/health", "/docs", "/upload", "POST /predict (multipart form-data field 'file')"],
    }

@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return """
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*"/>
      <button type="submit">Predict</button>
    </form>
    """

# ---------- Load classes & model ----------
with open("classes.json", "r", encoding="utf-8") as f:
    CLASSES = json.load(f)

ck = torch.load("best_model.pth", map_location="cpu", weights_only=True)
state = ck.get("state", ck)

# đoán kiến trúc từ key trong state_dict
if any(k.startswith("fc.") for k in state):
    arch = "resnet18"
elif any(k.startswith("classifier.") for k in state):
    arch = "mobilenet_v2"
else:
    arch = "cnn_cbam"  # demo này không phục vụ custom CNN trên server

if arch == "resnet18":
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
elif arch == "mobilenet_v2":
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASSES))
else:
    raise RuntimeError("Use server only for resnet/mobilenet in this demo.")

m.load_state_dict(state, strict=False)
m.eval()

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------- API ----------
@app.get("/health")
def health():
    return {"status": "ok", "arch": arch, "n_classes": len(CLASSES)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # nhận ảnh, convert RGB, transform
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = TF(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(m(x), dim=1).cpu().numpy()[0].tolist()

    k = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "pred": CLASSES[k],
        "probs": {CLASSES[i]: float(probs[i]) for i in range(len(probs))},
    }

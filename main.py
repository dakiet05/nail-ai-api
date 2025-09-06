import io, json, torch, torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

with open("classes.json") as f: CLASSES = json.load(f)
ck = torch.load("best_model.pth", map_location="cpu", weights_only=True)
state = ck.get("state", ck)

# đoán kiến trúc
arch = "resnet18" if any(k.startswith("fc.") for k in state) else ("mobilenet_v2" if any(k.startswith("classifier.") for k in state) else "cnn_cbam")
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
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@app.get("/health")
def health(): return {"status":"ok","arch":arch,"n_classes":len(CLASSES)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = TF(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(m(x),1).numpy()[0].tolist()
    k = int(max(range(len(p)), key=lambda i: p[i]))
    return {"pred": CLASSES[k], "probs": {CLASSES[i]: float(p[i]) for i in range(len(p))}}

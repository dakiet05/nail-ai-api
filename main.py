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

# ---------- UI landing page ----------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Nail AI – API Demo</title>
<style>
  :root { --bg:#0b1220; --card:#111a2b; --muted:#9aa0a6; --text:#e6edf3; --acc:#4C78A8; }
  body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,Segoe UI,Roboto}
  .wrap{max-width:960px;margin:auto;padding:24px}
  .hero{display:flex;align-items:center;gap:18px;margin:8px 0 20px}
  .hero h1{margin:0;font-size:28px}
  .card{background:var(--card);border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:20px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
  .grid{display:grid;gap:16px;grid-template-columns:1.2fr .8fr}
  @media (max-width:800px){.grid{grid-template-columns:1fr}}
  .drop{border:2px dashed rgba(255,255,255,.15);border-radius:14px;padding:18px;text-align:center;cursor:pointer;transition:.2s}
  .drop:hover{border-color:var(--acc)}
  .btn{background:var(--acc);color:white;border:0;border-radius:10px;padding:10px 16px;font-weight:600;cursor:pointer}
  .btn[disabled]{opacity:.5;cursor:default}
  .muted{color:var(--muted)}
  img.preview{max-width:100%;max-height:320px;border-radius:12px;display:block;margin:auto}
  .row{display:flex;justify-content:space-between;gap:10px}
  .bar{height:10px;background:#222b3d;border-radius:8px;overflow:hidden}
  .bar>i{display:block;height:100%;background:var(--acc)}
  .pill{display:inline-block;padding:4px 10px;border-radius:999px;background:#1f2a40}
  .footer{margin-top:16px;color:var(--muted);font-size:13px}
  a{color:#7fb1ff;text-decoration:none}
</style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <svg width="34" height="34" viewBox="0 0 24 24" fill="none"><path d="M12 2l3 7h7l-5.5 4 2 7-6.5-4.5L5.5 20l2-7L2 9h7l3-7z" stroke="#4C78A8" stroke-width="1.3" fill="none"/></svg>
      <h1>Nail AI – Demo API</h1>
    </div>

    <div class="card grid">
      <div>
        <div id="drop" class="drop">
          <input id="file" type="file" accept="image/*" style="display:none" />
          <p><b>Kéo-thả</b> ảnh vào đây hoặc <u>nhấn chọn ảnh</u></p>
          <p class="muted">Hỗ trợ .jpg/.png. Ảnh móng tay càng rõ càng tốt.</p>
        </div>
        <div id="previewWrap" style="display:none;margin-top:14px">
          <img id="preview" class="preview" alt="preview"/>
        </div>
        <div style="margin-top:14px;display:flex;gap:10px">
          <button id="btnPredict" class="btn" disabled>Dự đoán</button>
          <a class="pill" href="/docs" target="_blank">/docs</a>
          <a class="pill" href="/health" target="_blank">/health</a>
        </div>
        <div id="msg" class="footer"></div>
      </div>

      <div>
        <h3 style="margin:0 0 10px">Kết quả</h3>
        <div id="result" class="muted">Chưa có kết quả. Chọn ảnh rồi bấm <b>Dự đoán</b>.</div>
      </div>
    </div>

    <p class="footer">API endpoint: <code>POST /predict</code> (multipart/form-data field <code>file</code>).</p>
  </div>

<script>
const el=(id)=>document.getElementById(id);
const fileInput=el('file'),drop=el('drop'),preview=el('preview'),previewWrap=el('previewWrap'),btn=el('btnPredict'),result=el('result'),msg=el('msg');
const API='/predict';
function setMsg(t){msg.textContent=t||''}
function humanProb(p){return(p*100).toFixed(1)+'%'}
function renderResult(json){
  const entries=Object.entries(json.probs||{}).sort((a,b)=>b[1]-a[1]);
  let html='';
  if(entries.length){
    const best=entries[0];
    html+=`<div style="margin-bottom:8px">Dự đoán: <b>${json.pred}</b> <span class="pill">${humanProb(best[1])}</span></div>`;
  }
  html+=entries.map(([k,v])=>`
    <div>
      <div class="row"><span>${k}</span><span>${humanProb(v)}</span></div>
      <div class="bar"><i style="width:${(v*100).toFixed(1)}%"></i></div>
    </div>`).join('');
  result.innerHTML=html||'<span class="muted">Không có dữ liệu.</span>';
}
function setPreview(f){const r=new FileReader();r.onload=e=>{preview.src=e.target.result;previewWrap.style.display='block';btn.disabled=false};r.readAsDataURL(f)}
drop.onclick=()=>fileInput.click();
drop.ondragover=e=>{e.preventDefault();drop.style.borderColor='#4C78A8'};
drop.ondragleave=()=>drop.style.borderColor='rgba(255,255,255,.15)';
drop.ondrop=e=>{e.preventDefault();drop.style.borderColor='rgba(255,255,255,.15)';const f=e.dataTransfer.files?.[0];if(f){fileInput.files=e.dataTransfer.files;setPreview(f)}}
fileInput.onchange=e=>{const f=e.target.files?.[0];if(f)setPreview(f)}
btn.onclick=async()=>{
  const f=fileInput.files?.[0];if(!f){setMsg('Chưa chọn ảnh');return}
  btn.disabled=true;setMsg('Đang dự đoán...');
  try{
    const fd=new FormData();fd.append('file',f,'nail.jpg');
    const res=await fetch(API,{method:'POST',body:fd});
    if(!res.ok)throw new Error('HTTP '+res.status);
    renderResult(await res.json());setMsg('Xong!')
  }catch(err){result.innerHTML=`<span style="color:#ef4444">${err}</span>`;setMsg('')}finally{btn.disabled=false}
}
</script>
</body>
</html>
    """

@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    # redirect về trang chủ
    return """<script>location.href='/'</script>"""

# ---------- Load model ----------
with open("classes.json") as f:
    CLASSES = json.load(f)

ck = torch.load("best_model.pth", map_location="cpu", weights_only=True)
state = ck.get("state", ck)

if any(k.startswith("fc.") for k in state):
    arch = "resnet18"
elif any(k.startswith("classifier.") for k in state):
    arch = "mobilenet_v2"
else:
    arch = "cnn_cbam"

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
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = TF(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(m(x), 1).numpy()[0].tolist()
    k = int(max(range(len(p)), key=lambda i: p[i]))
    return {"pred": CLASSES[k], "probs": {CLASSES[i]: float(p[i]) for i in range(len(p))}}

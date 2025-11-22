from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import io
from PIL import Image
import torch

from ml.inference import (
    get_device,
    load_model,
    get_inference_transform,
    DEFAULT_MODEL_NAME,
)

app = FastAPI(
    title="WasteVision API",
    description="Backend API for WasteVision waste image classification",
    version="1.0.0",
)

# --- CORS (biar Flutter / tools lain gampang akses) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # untuk dev; nanti bisa dibatasi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global objects (dipakai ulang) ---
device = get_device()
model = None
class_names = None
transform = get_inference_transform(img_size=224)


class PredictionResponse(BaseModel):
    label: str
    confidence: float


@app.on_event("startup")
def load_model_on_startup():
    """
    Dipanggil sekali ketika server FastAPI start.
    Kita load model + class_names ke memori supaya
    tidak load ulang di setiap request /predict.
    """
    global model, class_names
    print("Loading model on startup...")
    m, classes = load_model(DEFAULT_MODEL_NAME, device)
    model = m
    class_names = classes
    print(f"Model loaded: {DEFAULT_MODEL_NAME} with {len(class_names)} classes.")


@app.get("/health")
def health_check():
    """
    Endpoint sederhana untuk cek apakah API & model siap.
    Bisa dipakai juga untuk health check di hosting.
    """
    return {
        "status": "ok",
        "model": DEFAULT_MODEL_NAME,
        "num_classes": len(class_names) if class_names else None,
        "device": str(device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Menerima file gambar (multipart/form-data) dan
    mengembalikan label + confidence.

    Ini yang nanti akan dipanggil dari Flutter:
    - kirim file image
    - terima JSON { label, confidence }
    """
    if model is None or class_names is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Validasi tipe file
    if file.content_type not in {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}",
        )

    # Baca bytes file
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    # Preprocessing → tensor
    tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = class_names[pred_idx.item()]
    confidence = float(conf.item())

    return PredictionResponse(label=label, confidence=confidence)
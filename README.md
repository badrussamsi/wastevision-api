# 🗑️ WasteVision — AI-Powered Waste Classification (Flutter + FastAPI + PyTorch)

**WasteVision** is an end-to-end AI system that identifies waste categories from images in real time.  
Built for mobile, optimized for CPU inference, and deployed fully on the cloud.

This project showcases:
- Mobile development (Flutter)
- Machine Learning (PyTorch)
- Production-ready API design (FastAPI + Docker)
- Cloud deployment (Render)
- Model training lifecycle (v2.0 → v2.2)

---

<div align="center">
  <img src="https://img.shields.io/badge/Flutter-Mobile-blue" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-brightgreen" />
  <img src="https://img.shields.io/badge/PyTorch-ML%20Model-orange" />
  <img src="https://img.shields.io/badge/Docker-Deploy-blue" />
  <img src="https://img.shields.io/badge/Render-Cloud-purple" />
</div>

---

# 🎯 Vision
> Make waste classification **accessible**, **accurate**, and **mobile‑friendly**, using lightweight ML models optimized for real-world conditions.

---

# 🧠 Model Summary (WasteVision v2.2)

| Metric | Result |
|-------|--------|
| **Architecture** | MobileNetV2 (ImageNet pretrained) |
| **Classes** | 7 (Cardboard, Organics, Glass, Metal, Misc, Paper, Plastic) |
| **Validation Accuracy** | **95.49%** |
| **Real-world Plastic Sachet** | **113 / 114 correct** |
| **Model Size** | ~17MB (optimized CPU inference) |

### ✔ Strengths
- High accuracy on plastic, paper, and glass  
- Robust to lighting & background variations  
- Very fast inference even on CPU-only cloud  

### ⚠️ Known Challenges
- Glossy plastic may resemble metal  
- Harsh shadows reduce confidence  

---

# 📸 Real‑World Test Results (Sample)

| Input | Expected | Predicted | Confidence |
|-------|----------|-----------|------------|
| Plastic sachet | Plastic | Plastic | **99.8%** |
| Plastic wrap | Plastic | Plastic | **98.6%** |
| Metal can | Metal | Metal | **99.8%** |
| Paper | Paper | Paper | **100%** |

---

# 🏛 System Architecture (End-to-End)

```
Flutter App (Camera/Gallery)
          |
          v
FastAPI Backend  -->  PyTorch Model (MobileNetV2 v2.2)
(Docker + Render)
          |
          v
   JSON Prediction
```

---

# 📂 Repository Overview (API)

```
wastevision-api/
│
├── app/
│   └── main.py              # API entrypoint, routes, health/ready checks
│
├── ml/
│   ├── inference.py         # CPU/MPS inference pipeline
│   ├── train_v2.py          # Training loop for v2.x
│   ├── config_v2.py
│   ├── datasets.py
│   └── tools/               # dataset merge, augmentation, debugging
│
├── models/
│   ├── wastevision_v2_2.pth
│   ├── wastevision_v2_2_classes.json
│   └── archive/             # older models (v2.0, v2.1)
│
├── Dockerfile
└── requirements.txt
```

---

# 📚 Dataset Sources (Acknowledgements)

The model was trained using a curated combination of public datasets from Kaggle.  
All datasets remain the property of their respective creators.

- **Garbage Classification Dataset**  
  https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification  

- **RealWaste Dataset**  
  https://www.kaggle.com/datasets/joebeachcapital/realwaste  

- **Trash Type Image Dataset**  
  https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset  

- **Waste Image Data**  
  https://www.kaggle.com/datasets/alveddian/waste-image-data  

Additional real‑world samples (114 plastic sachet images) were added to improve robustness on hard cases.

---

# 🚀 Local Development

## 1️⃣ Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2️⃣ Run API
```bash
uvicorn app.main:app --reload --port 8000
```

---

# 🐳 Docker Deployment

## Build image
```bash
docker build -t wastevision-api-local .
```

## Run container
```bash
docker run -p 8000:8000 wastevision-api-local
```

---

# 🌐 Cloud Deployment (Render)

- Fully Dockerized  
- CPU-only runtime → compatible with free tier  
- Startup loads model → readiness probe ensures stability  

### Health & Status
- `/health` → model + system info  
- `/ready` → readiness (model loaded)  
- `/predict` → main classification API  

---

# 🧪 API Reference

## ✔ GET `/health`
```json
{
  "status": "ok",
  "model": "wastevision_v2_2",
  "num_classes": 7,
  "device": "cpu"
}
```

## ✔ GET `/ready`
```json
{ "ready": true }
```

## ✔ POST `/predict`
```bash
curl -X POST \
  -F "file=@example.jpg" \
  http://localhost:8000/predict
```

Response:
```json
{
  "label": "Plastic",
  "confidence": 0.9862
}
```

---

# 🌱 Training Pipeline (v2.x)

1. Dataset merge & cleanup  
2. Train/val split (stratified)  
3. Training on Apple MPS (local GPU)  
4. Best-epoch checkpointing  
5. Model card documentation  
6. Release cycle: v2.0 → v2.1 → v2.2  

---

# 🔮 Roadmap

### v2.3 (Upcoming)
- Reintroduce 9-class dataset  
- Hard case mining via API  
- Improved augmentation for glossy plastic  
- Better plastic‑vs‑metal separation  

### Backend Enhancements
- GitHub Actions CI smoke tests  
- `/hardcase` endpoint  
- Improved structured logging  

### Mobile App
- Show confidence levels  
- Developer debug mode  
- Low-light detection  

---

# 👤 Author
Created as an AI/ML + Mobile + Cloud engineering portfolio project.

---

# 📄 License
MIT License
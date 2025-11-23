import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODELS_DIR = Path("models")
DEFAULT_MODEL_NAME = "wastevision_v2"


def get_device() -> torch.device:
    """
    Pilih device secara otomatis:
    - CUDA (NVIDIA) kalau ada
    - MPS (Apple Silicon) kalau ada
    - CPU kalau tidak ada GPU
    """
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal) GPU")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def load_class_names(model_name: str) -> list:
    classes_path = MODELS_DIR / f"{model_name}_classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"Class names file not found: {classes_path}")
    with open(classes_path, "r") as f:
        class_names = json.load(f)
    return class_names


def build_model(num_classes: int) -> nn.Module:
    """
    Bangun MobileNetV2 dengan head yang sama seperti waktu training.
    """
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(model_name: str, device: torch.device) -> Tuple[nn.Module, list]:
    """
    Load model dari file .pth dan class_names dari .json.
    """
    weight_path = MODELS_DIR / f"{model_name}.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")

    class_names = load_class_names(model_name)
    num_classes = len(class_names)

    model = build_model(num_classes)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, class_names


def get_inference_transform(img_size: int = 224):
    """
    Transformasi untuk inference.
    Harus SAMA dengan val_tfms di datasets.py:
    Resize -> ToTensor -> Normalize(Imagenet mean/std).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def predict_image(
    image_path: Path,
    model: nn.Module,
    class_names: list,
    device: torch.device,
    img_size: int = 224,
) -> Tuple[str, float]:
    """
    Lakukan inference pada 1 gambar.
    Return: (label, confidence)
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")

    transform = get_inference_transform(img_size)
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = class_names[pred_idx.item()]
    confidence = conf.item()
    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="WasteVision single-image inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path ke file gambar (jpg/png) yang akan diprediksi",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Nama model (default: wastevision_v1)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)

    device = get_device()
    model, class_names = load_model(args.model_name, device)

    print(f"Loaded model '{args.model_name}' with {len(class_names)} classes.")
    print(f"Classes: {class_names}")
    print(f"Predicting image: {image_path}")

    label, confidence = predict_image(image_path, model, class_names, device)

    print("\n=== Prediction Result ===")
    print(f"Label      : {label}")
    print(f"Confidence : {confidence:.4f}")


if __name__ == "__main__":
    main()
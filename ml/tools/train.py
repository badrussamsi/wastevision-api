import json
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models

from .datasets import get_dataloaders


DATA_DIR = "/Users/omg/Code/playground/Datasets/RealWaste"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def get_device() -> torch.device:
    """
    Pilih device secara otomatis:
    - GPU (cuda/mps) kalau tersedia
    - fallback ke CPU kalau tidak ada
    """
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    # Support Apple Silicon (M1/M2) pakai Metal
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal) GPU")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    """
    Bangun model MobileNetV2 pre-trained di ImageNet,
    lalu ganti classifier terakhir untuk jumlah kelas RealWaste.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Ganti classifier terakhir
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Latih 1 epoch:
    - hitung loss rata-rata
    - hitung accuracy rata-rata
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluasi di validation set:
    - loss rata-rata
    - accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def save_model(model: nn.Module, class_names, model_name: str = "wastevision_v1"):
    """
    Simpan weight model + class names untuk inference nanti.
    """
    weight_path = MODELS_DIR / f"{model_name}.pth"
    classes_path = MODELS_DIR / f"{model_name}_classes.json"

    torch.save(model.state_dict(), weight_path)
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved model weights to: {weight_path}")
    print(f"Saved class names to:   {classes_path}")


def main():
    device = get_device()

    print("Loading dataloaders...")
    train_loader, val_loader, class_names = get_dataloaders(
        root_dir=DATA_DIR,
        batch_size=32,
        train_ratio=0.8,
        img_size=224,
    )

    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    num_epochs = 5  # bisa dinaikkan nanti kalau sudah oke

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Train  - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}\n"
            f"Val    - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}"
        )

        # Simpan model terbaik (berdasarkan val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best val acc: {best_val_acc:.4f} → saving model...")
            save_model(model, class_names, model_name="wastevision_v1")

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed/60:.2f} minutes.")
    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from ml import config_v2


def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def get_dataloaders():
    # Augmentasi untuk train
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(config_v2.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Val: hanya resize + center crop
    val_transforms = transforms.Compose([
        transforms.Resize(int(config_v2.IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(config_v2.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(
        root=config_v2.TRAIN_DIR,
        transform=train_transforms,
    )
    val_dataset = datasets.ImageFolder(
        root=config_v2.VAL_DIR,
        transform=val_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_v2.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config_v2.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Classes       : {config_v2.CLASS_NAMES} "
          f"({config_v2.NUM_CLASSES} classes)")

    return train_loader, val_loader


def build_model(num_classes: int):
    # Coba API baru dulu, kalau error fallback ke API lama
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT  # type: ignore[attr-defined]
        model = models.mobilenet_v2(weights=weights)
        print("Using MobileNet_V2 with DEFAULT weights")
    except Exception:
        model = models.mobilenet_v2(pretrained=True)
        print("Using MobileNet_V2 with pretrained=True (legacy API)")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        running_correct += preds.eq(labels).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    torch.set_grad_enabled(False)
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        running_correct += preds.eq(labels).sum().item()
        total += inputs.size(0)
    torch.set_grad_enabled(True)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def main():
    print("=== Training WasteVision V2 ===")
    print(f"Model version : {config_v2.MODEL_VERSION}")
    print(f"Description   : {config_v2.DESCRIPTION}")
    print(f"Seed          : {config_v2.SEED}")

    torch.manual_seed(config_v2.SEED)

    device = get_device()
    train_loader, val_loader = get_dataloaders()

    model = build_model(config_v2.NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config_v2.LEARNING_RATE,
        weight_decay=config_v2.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # berdasarkan val_acc
        factor=0.5,
        patience=2,
        #verbose=True,
    )

    best_val_acc = 0.0

    print(f"Output dir: {config_v2.OUTPUT_DIR}")
    config_v2.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, config_v2.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config_v2.EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        # Save last model
        torch.save(model.state_dict(), config_v2.CHECKPOINT_PATH)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config_v2.BEST_MODEL_PATH)
            print(f"✅ New best model saved with val_acc={best_val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed/60:.1f} minutes.")
    print(f"Best val acc: {best_val_acc:.4f}")

    # Save class mapping
    config_v2.save_class_mapping()
    print("Done.")


if __name__ == "__main__":
    main()
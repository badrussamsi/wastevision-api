import os
from typing import Tuple, List
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(img_size: int = 224):
    """
    Define image transformations for training and validation.
    Includes basic augmentation on training.
    """
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_tfms, val_tfms


def load_dataset(root_dir: str, img_size: int = 224):
    """
    Load dataset from folder. Folder must have structure:
    root_dir/
      ├── Class1/
      ├── Class2/
      ├── ...
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"Dataset folder not found: {root_dir}")

    train_tfms, val_tfms = get_transforms(img_size)

    # torchvision ImageFolder automatically maps subfolders → labels
    dataset = datasets.ImageFolder(root=root_dir, transform=None)

    # Save class names
    class_names = dataset.classes

    return dataset, class_names, train_tfms, val_tfms


def split_dataset(dataset, train_ratio: float = 0.8):
    """
    Split dataset into train and validation sets.
    """
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Public function used by train.py to load dataloaders.
    """
    dataset, class_names, train_tfms, val_tfms = load_dataset(root_dir, img_size)
    train_ds, val_ds = split_dataset(dataset, train_ratio)

    # Apply different transforms for train/val
    train_ds.dataset.transform = train_tfms
    val_ds.dataset.transform = val_tfms

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names
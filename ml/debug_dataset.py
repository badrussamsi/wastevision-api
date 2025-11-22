from .datasets import get_dataloaders

DATASET_PATH = "/Users/omg/Code/playground/Datasets/RealWaste"  # <--- pastikan sesuai folder kamu

def main():
    print("Loading dataset...")
    train_loader, val_loader, class_names = get_dataloaders(
        root_dir=DATASET_PATH,
        batch_size=4,
        train_ratio=0.8,
        img_size=224
    )

    print("Classes:", class_names)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # Ambil 1 batch
    print("Fetching one batch...")
    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)  # Expected: torch.Size([4, 3, 224, 224])
    print("Labels:", labels)                    # Expected: tensor([...])

if __name__ == "__main__":
    main()
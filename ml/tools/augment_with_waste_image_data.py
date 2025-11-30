# ml/tools/augment_with_waste_image_data.py

from pathlib import Path
import shutil

# Lokasi dataset eksternal (Kaggle) DI LUAR repo
EXTERNAL_ROOT = Path("/Users/omg/Code/playground/Datasets/waste-image-data")

# Lokasi dataset train utama yang dipakai untuk training v2
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_TRAIN_ROOT = PROJECT_ROOT / "ml" / "datasets" / "realwaste_v2" / "train"

# Mapping kelas eksternal -> kelas WasteVision
CLASS_MAPPING = {
    "battery": "Miscellaneous Trash",
    "biological": "Food Organics",
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "plasticbag": "Plastic",  # penting buat foil / kantong plastik
}

# Ekstensi file image yang kita anggap valid
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    external_train = EXTERNAL_ROOT / "train"
    if not external_train.exists():
        raise SystemExit(f"External train dir not found: {external_train}")

    print(f"External train dir: {external_train}")
    print(f"Target train root : {TARGET_TRAIN_ROOT}")

    for src_class, dst_class in CLASS_MAPPING.items():
        src_dir = external_train / src_class
        if not src_dir.exists():
            print(f"[WARN] Source class dir not found, skip: {src_dir}")
            continue

        dst_dir = TARGET_TRAIN_ROOT / dst_class
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = [p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        print(f"[INFO] {src_class} -> {dst_class}: {len(files)} files")

        for idx, src_path in enumerate(files, start=1):
            # Supaya nama file unik dan jelas asalnya
            new_name = f"wasteimg_{src_class}_{src_path.stem}{src_path.suffix}"
            dst_path = dst_dir / new_name

            shutil.copy2(src_path, dst_path)

            if idx % 100 == 0:
                print(f"  copied {idx} files for class {src_class}")

    print("Done augmenting realwaste_v2/train with waste-image-data.")


if __name__ == "__main__":
    main()
import os
from pathlib import Path

# ============================
#  Project & Version Metadata
# ============================

MODEL_VERSION = "wastevision_v2"
DESCRIPTION = "Improved classifier trained using RealWaste_Enhanced (20k images) with Kaggle fusion."
SEED = 42  # Ensures reproducibility

# ============================
#  Dataset Locations
# ============================

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "datasets" / "realwaste_v2"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"

# ============================
#  Training Hyperparameters
# ============================

IMAGE_SIZE = 224            # Standard for ResNet/MobileNet
BATCH_SIZE = 16             # Safe for CPU/GPU Mac
EPOCHS = 15                 # Start small, can increase later
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

USE_AUGMENTATION = True

# ============================
#  Class Mapping
# ============================

CLASS_NAMES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)

# Save a reference file for inference
CLASS_MAPPING_PATH = BASE_DIR / f"class_mapping_{MODEL_VERSION}.txt"

# ============================
#  Checkpoints / Outputs
# ============================

OUTPUT_DIR = BASE_DIR / "models" / MODEL_VERSION
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "model.pth"
BEST_MODEL_PATH = OUTPUT_DIR / "model_best.pth"

# ============================
#  Utility
# ============================

def save_class_mapping():
    """Save class names for inference processing."""
    with open(CLASS_MAPPING_PATH, "w") as f:
        for c in CLASS_NAMES:
            f.write(c + "\n")

    print(f"Saved class mapping to {CLASS_MAPPING_PATH}")
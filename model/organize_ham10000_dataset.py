"""
HAM10000 Dataset Organizer
Organizes images from HAM10000 into train/val folders by skin condition.
"""
import os
import shutil
import random
import pandas as pd
from pathlib import Path

# --- CONFIG ---
EXTRACT_DIR = Path(__file__).parent / "data" / "ham10000"
IMG_DIR = EXTRACT_DIR
CSV_PATH = EXTRACT_DIR / "HAM10000_metadata.csv"
DATA_ROOT = Path(__file__).parent / "data"
TRAIN_RATIO = 0.7

# --- SKIN CONDITION LABELS ---
# These are the labels in HAM10000
LABELS = [
    "nv",  # melanocytic nevi
    "mel", # melanoma
    "bkl", # benign keratosis-like lesions
    "bcc", # basal cell carcinoma
    "akiec", # actinic keratoses
    "vasc", # vascular lesions
    "df"    # dermatofibroma
]

# --- CREATE FOLDERS ---
for split in ["train", "val"]:
    for cls in LABELS:
        Path(DATA_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

# --- ORGANIZE IMAGES ---
def organize_images():
    df = pd.read_csv(CSV_PATH)
    for cls in LABELS:
        images = df[df['dx'] == cls]['image_id'].tolist()
        random.shuffle(images)
        n_train = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:]
        for img_id in train_imgs:
            src = IMG_DIR / f"{img_id}.jpg"
            dst = DATA_ROOT / "train" / cls / f"{img_id}.jpg"
            if src.exists():
                shutil.copy(src, dst)
        for img_id in val_imgs:
            src = IMG_DIR / f"{img_id}.jpg"
            dst = DATA_ROOT / "val" / cls / f"{img_id}.jpg"
            if src.exists():
                shutil.copy(src, dst)
    print("Images organized into train/val folders by class.")

if __name__ == "__main__":
    organize_images()

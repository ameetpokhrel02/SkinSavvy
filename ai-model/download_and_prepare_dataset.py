"""
Skin Condition Dataset Downloader & Organizer
Downloads sample images for acne, dark spots, and wrinkles from open sources.
Organizes them into train/val folders for model training.
"""

import os
import shutil
import random
import requests
from pathlib import Path

# --- CONFIG ---
DATA_ROOT = Path(__file__).parent / "data"
CLASSES = ["acne", "dark_spots", "wrinkles"]
SAMPLE_URLS = {
    "acne": [
        "https://dermnetnz.org/assets/Uploads/acne-vulgaris-1.jpg",
        "https://dermnetnz.org/assets/Uploads/acne-vulgaris-2.jpg"
    ],
    "dark_spots": [
        "https://dermnetnz.org/assets/Uploads/melasma-1.jpg",
        "https://dermnetnz.org/assets/Uploads/melasma-2.jpg"
    ],
    "wrinkles": [
        "https://dermnetnz.org/assets/Uploads/wrinkles-1.jpg",
        "https://dermnetnz.org/assets/Uploads/wrinkles-2.jpg"
    ]
}
TRAIN_RATIO = 0.7

# --- CREATE FOLDERS ---
for split in ["train", "val"]:
    for cls in CLASSES:
        Path(DATA_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

# --- DOWNLOAD IMAGES ---
def download_images():
    for cls, urls in SAMPLE_URLS.items():
        for idx, url in enumerate(urls):
            ext = url.split('.')[-1]
            fname = f"{cls}_{idx}.{ext}"
            out_path = DATA_ROOT / "all" / cls
            out_path.mkdir(parents=True, exist_ok=True)
            img_path = out_path / fname
            if not img_path.exists():
                print(f"Downloading {url} -> {img_path}")
                try:
                    r = requests.get(url, timeout=10)
                    r.raise_for_status()
                    with open(img_path, "wb") as f:
                        f.write(r.content)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")

# --- SPLIT IMAGES ---
def split_images():
    for cls in CLASSES:
        all_dir = DATA_ROOT / "all" / cls
        images = list(all_dir.glob("*.jpg")) + list(all_dir.glob("*.jpeg")) + list(all_dir.glob("*.png"))
        random.shuffle(images)
        n_train = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:]
        for img in train_imgs:
            shutil.copy(img, DATA_ROOT / "train" / cls / img.name)
        for img in val_imgs:
            shutil.copy(img, DATA_ROOT / "val" / cls / img.name)

if __name__ == "__main__":
    download_images()
    split_images()
    print("Sample images downloaded and organized into train/val folders.")

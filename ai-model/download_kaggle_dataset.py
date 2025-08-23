"""
Kaggle Skin Disease Dataset Downloader
Downloads HAM10000 dataset from Kaggle and extracts images for training/validation.
"""
import os
import zipfile
from pathlib import Path
import subprocess

# --- CONFIG ---
DATASET = "andrewmvd/skin-cancer-mnist-ham10000"
DATA_DIR = Path(__file__).parent / "data"
ZIP_PATH = DATA_DIR / "ham10000.zip"
EXTRACT_DIR = DATA_DIR / "ham10000"

# --- DOWNLOAD DATASET ---
def download_kaggle_dataset():
    print("Downloading HAM10000 dataset from Kaggle...")
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR)]
    subprocess.run(cmd, check=True)
    print("Download complete.")

# --- EXTRACT ZIP ---
def extract_zip():
    print(f"Extracting {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Extraction complete.")

if __name__ == "__main__":
    if not ZIP_PATH.exists():
        download_kaggle_dataset()
    else:
        print(f"{ZIP_PATH} already exists.")
    if not EXTRACT_DIR.exists():
        extract_zip()
    else:
        print(f"{EXTRACT_DIR} already exists.")
    print("Dataset is ready. Please organize images into train/val folders as needed.")

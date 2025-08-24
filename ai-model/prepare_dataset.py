"""
Prepare dataset for binary classification
Organizes images into acne and no_acne folders based on annotations
"""

import pandas as pd
import os
import shutil
from pathlib import Path
import random

def prepare_dataset():
    # Paths
    base_dir = Path(__file__).parent.parent
    train_dir = base_dir / 'acne_test_dataset' / 'train'
    val_dir = base_dir / 'acne_test_dataset' / 'valid'
    
    # Create organized dataset directories
    organized_dir = base_dir / 'ai-model' / 'data_organized'
    train_organized = organized_dir / 'train'
    val_organized = organized_dir / 'val'
    
    # Create directories
    for split_dir in [train_organized / 'acne', train_organized / 'no_acne',
                     val_organized / 'acne', val_organized / 'no_acne']:
        split_dir.mkdir(parents=True, exist_ok=True)
    
    print("Organizing training data...")
    organize_split(train_dir, train_organized, 'train')
    
    print("Organizing validation data...")
    organize_split(val_dir, val_organized, 'val')
    
    print("Dataset preparation completed!")
    print(f"Organized data saved to: {organized_dir}")

def organize_split(source_dir, target_dir, split_name):
    """Organize images from source directory into acne/no_acne folders"""
    
    # Read annotations
    annotations_file = source_dir / '_annotations.csv'
    if not annotations_file.exists():
        print(f"Warning: No annotations file found in {source_dir}")
        return
    
    df = pd.read_csv(annotations_file)
    
    # Get unique filenames and their acne status
    image_acne_status = {}
    
    for filename in df['filename'].unique():
        file_annotations = df[df['filename'] == filename]
        # Check if any annotation has 'acne' class
        has_acne = any(file_annotations['class'].str.contains('acne', case=False))
        image_acne_status[filename] = has_acne
    
    # Copy images to appropriate folders
    for filename, has_acne in image_acne_status.items():
        source_path = source_dir / filename
        if source_path.exists():
            if has_acne:
                target_path = target_dir / 'acne' / filename
            else:
                target_path = target_dir / 'no_acne' / filename
            
            shutil.copy2(source_path, target_path)
    
    # Print statistics
    acne_count = len(list((target_dir / 'acne').glob('*.jpg')))
    no_acne_count = len(list((target_dir / 'no_acne').glob('*.jpg')))
    
    print(f"{split_name.capitalize()} set:")
    print(f"  Acne images: {acne_count}")
    print(f"  No acne images: {no_acne_count}")
    print(f"  Total: {acne_count + no_acne_count}")

if __name__ == "__main__":
    prepare_dataset()

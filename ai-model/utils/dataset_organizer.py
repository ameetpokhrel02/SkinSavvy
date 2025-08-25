"""
Dataset Organizer for Multi-Class Skin Condition Classification
Helps organize images into the required folder structure
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image
import argparse

class DatasetOrganizer:
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.classes = ['Acne', 'Pimple', 'Spots', 'Mole1', 'Mole2', 'Scar']
        
        # Create output directory structure
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create the required directory structure"""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            for class_name in self.classes:
                class_dir = self.output_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {class_dir}")
    
    def organize_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Organize dataset into train/val/test splits"""
        print(f"\n📁 Organizing dataset from: {self.source_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Verify ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        total_processed = 0
        
        for class_name in self.classes:
            print(f"\n🔄 Processing class: {class_name}")
            
            # Find source images for this class
            source_class_dir = self.source_dir / class_name
            if not source_class_dir.exists():
                print(f"⚠️  Warning: Source directory for {class_name} not found: {source_class_dir}")
                continue
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(source_class_dir.glob(f'*{ext}')))
                image_files.extend(list(source_class_dir.glob(f'*{ext.upper()}')))
            
            if not image_files:
                print(f"⚠️  Warning: No images found for {class_name}")
                continue
            
            print(f"📊 Found {len(image_files)} images for {class_name}")
            
            # Shuffle files for random split
            random.shuffle(image_files)
            
            # Calculate split indices
            total_images = len(image_files)
            train_end = int(total_images * train_ratio)
            val_end = train_end + int(total_images * val_ratio)
            
            # Split files
            train_files = image_files[:train_end]
            val_files = image_files[train_end:val_end]
            test_files = image_files[val_end:]
            
            print(f"  Train: {len(train_files)} images")
            print(f"  Val: {len(val_files)} images")
            print(f"  Test: {len(test_files)} images")
            
            # Copy files to appropriate directories
            self.copy_files(train_files, 'train', class_name)
            self.copy_files(val_files, 'val', class_name)
            self.copy_files(test_files, 'test', class_name)
            
            total_processed += total_images
        
        print(f"\n✅ Dataset organization completed!")
        print(f"📊 Total images processed: {total_processed}")
        self.print_dataset_summary()
    
    def copy_files(self, files, split, class_name):
        """Copy files to the appropriate directory"""
        target_dir = self.output_dir / split / class_name
        
        for file_path in files:
            try:
                # Copy file with original name
                shutil.copy2(file_path, target_dir / file_path.name)
            except Exception as e:
                print(f"⚠️  Error copying {file_path}: {e}")
    
    def print_dataset_summary(self):
        """Print summary of organized dataset"""
        print(f"\n📊 Dataset Summary:")
        print("=" * 50)
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            print(f"\n{split.upper()} SET:")
            total_split = 0
            
            for class_name in self.classes:
                class_dir = self.output_dir / split / class_name
                if class_dir.exists():
                    image_count = len(list(class_dir.glob('*.jpg')) + 
                                    list(class_dir.glob('*.jpeg')) + 
                                    list(class_dir.glob('*.png')) + 
                                    list(class_dir.glob('*.bmp')) + 
                                    list(class_dir.glob('*.tiff')))
                    print(f"  {class_name}: {image_count} images")
                    total_split += image_count
                else:
                    print(f"  {class_name}: 0 images (directory not found)")
            
            print(f"  Total: {total_split} images")
    
    def validate_images(self):
        """Validate that all images can be opened"""
        print(f"\n🔍 Validating images...")
        
        invalid_files = []
        total_files = 0
        
        for split in ['train', 'val', 'test']:
            for class_name in self.classes:
                class_dir = self.output_dir / split / class_name
                if not class_dir.exists():
                    continue
                
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + \
                             list(class_dir.glob('*.png')) + list(class_dir.glob('*.bmp')) + \
                             list(class_dir.glob('*.tiff'))
                
                for image_file in image_files:
                    total_files += 1
                    try:
                        with Image.open(image_file) as img:
                            img.verify()
                    except Exception as e:
                        invalid_files.append((image_file, str(e)))
        
        print(f"📊 Validation complete:")
        print(f"  Total files checked: {total_files}")
        print(f"  Valid files: {total_files - len(invalid_files)}")
        print(f"  Invalid files: {len(invalid_files)}")
        
        if invalid_files:
            print(f"\n⚠️  Invalid files found:")
            for file_path, error in invalid_files[:10]:  # Show first 10
                print(f"  {file_path}: {error}")
            if len(invalid_files) > 10:
                print(f"  ... and {len(invalid_files) - 10} more")
    
    def create_sample_dataset(self, samples_per_class=50):
        """Create a small sample dataset for testing"""
        print(f"\n🧪 Creating sample dataset with {samples_per_class} samples per class...")
        
        sample_dir = self.output_dir.parent / 'sample_dataset'
        
        for split in ['train', 'val', 'test']:
            for class_name in self.classes:
                source_dir = self.output_dir / split / class_name
                target_dir = sample_dir / split / class_name
                target_dir.mkdir(parents=True, exist_ok=True)
                
                if source_dir.exists():
                    image_files = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.jpeg')) + \
                                 list(source_dir.glob('*.png')) + list(source_dir.glob('*.bmp')) + \
                                 list(source_dir.glob('*.tiff'))
                    
                    # Take random samples
                    sample_files = random.sample(image_files, min(samples_per_class, len(image_files)))
                    
                    for file_path in sample_files:
                        shutil.copy2(file_path, target_dir / file_path.name)
        
        print(f"✅ Sample dataset created at: {sample_dir}")

def main():
    parser = argparse.ArgumentParser(description='Organize skin condition dataset')
    parser.add_argument('--source', required=True, help='Source directory containing class folders')
    parser.add_argument('--output', required=True, help='Output directory for organized dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio (default: 0.1)')
    parser.add_argument('--validate', action='store_true', help='Validate images after organization')
    parser.add_argument('--sample', type=int, help='Create sample dataset with N samples per class')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Create organizer
    organizer = DatasetOrganizer(args.source, args.output)
    
    # Organize dataset
    organizer.organize_dataset(args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Validate images if requested
    if args.validate:
        organizer.validate_images()
    
    # Create sample dataset if requested
    if args.sample:
        organizer.create_sample_dataset(args.sample)

if __name__ == '__main__':
    main()

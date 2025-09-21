#!/usr/bin/env python3
"""
Verify that the dataset directories specified in data.yaml exist.
"""

import os
import yaml
from pathlib import Path

def main():
    """Main function to verify dataset paths."""
    # Load data.yaml file
    try:
        with open('data.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        print("Successfully loaded data.yaml")
    except Exception as e:
        print(f"Error loading data.yaml: {e}")
        return
    
    # Get paths from config
    train_images_dir = Path(data_config['train'])
    val_images_dir = Path(data_config['val'])
    labels_dir = Path(data_config.get('labels', ''))
    
    # Check if paths exist
    print("\nVerifying paths...")
    
    # Check training images directory
    if train_images_dir.exists():
        num_train_images = len([f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✓ Train images directory exists: {train_images_dir}")
        print(f"  - Contains {num_train_images} images")
        
        # List a few sample files
        if num_train_images > 0:
            sample_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:3]
            print(f"  - Sample files: {', '.join(sample_files)}")
    else:
        print(f"✕ Train images directory does not exist: {train_images_dir}")
    
    # Check validation images directory
    if val_images_dir.exists():
        num_val_images = len([f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✓ Validation images directory exists: {val_images_dir}")
        print(f"  - Contains {num_val_images} images")
    else:
        print(f"✕ Validation images directory does not exist: {val_images_dir}")
    
    # Check labels directory
    if labels_dir.exists():
        num_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
        print(f"✓ Labels directory exists: {labels_dir}")
        print(f"  - Contains {num_labels} label files")
    else:
        print(f"✕ Labels directory does not exist: {labels_dir}")
    
    # Check a sample image and label
    if train_images_dir.exists() and labels_dir.exists():
        print("\nVerifying sample files...")
        try:
            image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                sample_image = image_files[0]
                sample_label = sample_image.rsplit('.', 1)[0] + '.txt'
                
                sample_image_path = train_images_dir / sample_image
                sample_label_path = labels_dir / sample_label
                
                if sample_image_path.exists():
                    print(f"✓ Sample image exists: {sample_image_path}")
                    print(f"  - Size: {os.path.getsize(sample_image_path) / 1024:.1f} KB")
                else:
                    print(f"✕ Sample image does not exist: {sample_image_path}")
                
                if sample_label_path.exists():
                    print(f"✓ Sample label exists: {sample_label_path}")
                    with open(sample_label_path, 'r') as f:
                        num_lines = len(f.readlines())
                    print(f"  - Contains {num_lines} annotation(s)")
                else:
                    print(f"✕ Sample label does not exist: {sample_label_path}")
        except Exception as e:
            print(f"Error verifying sample files: {e}")
    
    print("\nSummary:")
    if train_images_dir.exists() and labels_dir.exists():
        print("✓ Dataset appears to be valid.")
        print("\nYou can now run the dataset splitting script:")
        print("   split_dataset.bat")
    else:
        print("✕ Dataset appears to be invalid. Please check the paths in data.yaml.")

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
Split the PPE detection dataset into training and validation sets.
70% for training and 30% for validation.
"""

import os
import shutil
import random
import yaml
from pathlib import Path
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Split dataset into training and validation sets')
    
    parser.add_argument('--data_yaml', type=str, default='data.yaml',
                        help='Path to data.yaml file')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def create_directory(directory):
    """Create directory if it doesn't exist."""
    dir_path = Path(directory)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    return dir_path

def main():
    """Main function to split the dataset."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load data.yaml file
    try:
        with open(args.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading data.yaml: {e}")
        return
    
    # Get paths from config
    original_images_dir = Path(data_config['train'])
    original_labels_dir = Path(data_config.get('labels', str(original_images_dir).replace('images', 'labels')))
    
    if not original_images_dir.exists():
        print(f"Error: Images directory '{original_images_dir}' does not exist")
        return
    
    if not original_labels_dir.exists():
        print(f"Error: Labels directory '{original_labels_dir}' does not exist")
        return
    
    # Create dataset directory structure
    dataset_dir = Path('dataset')
    train_images_dir = create_directory(dataset_dir / 'train' / 'images')
    train_labels_dir = create_directory(dataset_dir / 'train' / 'labels')
    val_images_dir = create_directory(dataset_dir / 'val' / 'images')
    val_labels_dir = create_directory(dataset_dir / 'val' / 'labels')
    
    # Get list of all image files
    image_files = [f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {original_images_dir}")
        return
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Split into train and validation sets
    split_idx = int(len(image_files) * args.train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"Validation images: {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)")
    
    # Copy training files
    for img_file in train_files:
        # Copy image
        src_img = original_images_dir / img_file
        dst_img = train_images_dir / img_file
        shutil.copy2(src_img, dst_img)
        
        # Copy corresponding label file
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        src_label = original_labels_dir / label_file
        if src_label.exists():
            dst_label = train_labels_dir / label_file
            shutil.copy2(src_label, dst_label)
    
    # Copy validation files
    for img_file in val_files:
        # Copy image
        src_img = original_images_dir / img_file
        dst_img = val_images_dir / img_file
        shutil.copy2(src_img, dst_img)
        
        # Copy corresponding label file
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        src_label = original_labels_dir / label_file
        if src_label.exists():
            dst_label = val_labels_dir / label_file
            shutil.copy2(src_label, dst_label)
    
    # Update data.yaml with new paths
    data_config['train'] = str(train_images_dir)
    data_config['val'] = str(val_images_dir)
    data_config['labels'] = str(train_labels_dir).replace('train', '')  # Parent labels directory
    
    # Save updated data.yaml
    new_data_yaml = 'data_split.yaml'
    with open(new_data_yaml, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"\nDataset successfully split!")
    print(f"Updated configuration saved to: {new_data_yaml}")
    print("\nTo use the split dataset for training, run:")
    print(f"python train_ppe_model.py --data {new_data_yaml}")

if __name__ == '__main__':
    main() 
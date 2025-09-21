#!/usr/bin/env python3
"""
Train a YOLO model for PPE detection using the downloaded dataset.
"""

import os
import argparse
import yaml
import torch
import time
from ultralytics import YOLO
from pathlib import Path
import shutil
import logging
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_user_input(prompt, default=None):
    """Get user input with optional default value."""
    if default is not None:
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default
    return input(f"{prompt}: ").strip()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLO model for PPE detection')
    
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img', type=int, default=640,
                        help='Image size')
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='Initial weights path')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project name')
    parser.add_argument('--name', type=str, default=f'exp_{int(time.time())}',
                        help='Experiment name (default: timestamped experiment name)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Optimizer (SGD, Adam, AdamW)')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final learning rate (fraction of lr0)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--save-period', type=int, default=10,
                        help='Save checkpoint every x epochs')
    parser.add_argument('--exist-ok', action='store_true',
                        help='Overwrite existing experiment directory')
    parser.add_argument('--cos-lr', action='store_true',
                        help='Use cosine learning rate scheduler')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic augmentation (0.0 to 1.0)')
    parser.add_argument('--mixup', type=float, default=0.05,
                        help='Mixup augmentation (0.0 to 1.0)')
    parser.add_argument('--copy-paste', type=float, default=0.1,
                        help='Copy-paste augmentation (0.0 to 1.0)')
    parser.add_argument('--translate', type=float, default=0.2,
                        help='Translation augmentation (0.0 to 1.0)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scaling augmentation (0.0 to 1.0)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='Horizontal flip augmentation (0.0 to 1.0)')
    parser.add_argument('--degrees', type=float, default=10.0,
                        help='Rotation augmentation in degrees')
    
    return parser.parse_args()

def validate_data_yaml(data_path):
    """Validate that the data.yaml file exists and has the right structure."""
    try:
        data_path = Path(data_path)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return False
            
        with open(data_path, 'r') as f:
            data = yaml.safe_load(f)
            
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required key '{key}' in {data_path}")
                return False
                
        # Check if train and val paths exist
        train_path = Path(data['train'])
        if not train_path.exists():
            logger.error(f"Train path not found: {train_path}")
            return False
            
        val_path = Path(data['val'])
        if not val_path.exists():
            logger.error(f"Validation path not found: {val_path}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating data.yaml: {e}")
        return False

def prepare_directories(project, name, exist_ok=False):
    """Prepare directories for training."""
    # Create project directory
    project_dir = Path(project)
    project_dir.mkdir(exist_ok=True, parents=True)
    
    # Create experiment directory
    exp_dir = project_dir / name
    if exp_dir.exists() and not exist_ok:
        logger.info(f"Experiment directory already exists: {exp_dir}")
        logger.info("Creating a timestamped experiment directory instead.")
        name = f"exp_{int(time.time())}"
        exp_dir = project_dir / name
    
    exp_dir.mkdir(exist_ok=exist_ok, parents=True)
    logger.info(f"Created experiment directory: {exp_dir}")
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    return name

def main():
    """Main function."""
    args = parse_args()
    
    # Validate data.yaml
    if not validate_data_yaml(args.data):
        logger.error("Invalid data.yaml file. Please check the file and try again.")
        return
    
    # Create directories
    name = prepare_directories(args.project, args.name, args.exist_ok)
    if name != args.name:
        args.name = name
    
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"CUDA is available with {num_gpus} GPU(s)")
        logger.info(f"Current device: {current_device} - {device_name}")
        
        # Set device to GPU if available
        if args.device == '':
            args.device = '0'  # Default to first GPU
        
        # Check if specified device is valid
        if args.device != 'cpu' and not args.device.isdigit() and not all(d.isdigit() for d in args.device.split(',')):
            logger.warning(f"Invalid device '{args.device}'. Using default device '0'.")
            args.device = '0'
    else:
        logger.warning("CUDA is not available. Training will use CPU, which may be slow.")
        args.device = 'cpu'
    
    # Print training configuration
    logger.info("\n=== Training Configuration ===")
    logger.info(f"Data: {args.data}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch}")
    logger.info(f"Image size: {args.img}")
    logger.info(f"Initial weights: {args.weights}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Project: {args.project}")
    logger.info(f"Experiment name: {args.name}")
    logger.info(f"Optimizer: {args.optimizer}")
    logger.info(f"Initial learning rate: {args.lr0}")
    logger.info(f"Final learning rate: {args.lrf}")
    logger.info(f"Learning rate schedule: {'Cosine' if args.cos_lr else 'Linear'}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Data augmentation: {'Yes' if args.augment else 'No'}")
    if args.augment:
        logger.info(f"  Mosaic: {args.mosaic}")
        logger.info(f"  Mixup: {args.mixup}")
        logger.info(f"  Copy-paste: {args.copy_paste}")
        logger.info(f"  Translation: {args.translate}")
        logger.info(f"  Scale: {args.scale}")
        logger.info(f"  Horizontal flip: {args.fliplr}")
        logger.info(f"  Rotation: {args.degrees} degrees")
    logger.info(f"Overwrite existing: {'Yes' if args.exist_ok else 'No'}")
    logger.info("=============================\n")
    
    # Confirm training configuration
    response = get_user_input("Do you want to proceed with this configuration? (yes/no)", default="yes").lower()
    if response != "yes":
        logger.info("Training cancelled.")
        return
    
    try:
        # Load model
        logger.info(f"Loading model from {args.weights}")
        model = YOLO(args.weights)
        
        # Prepare augmentation settings
        augment_params = {}
        if args.augment:
            augment_params = {
                'mosaic': args.mosaic,
                'mixup': args.mixup,
                'copy_paste': args.copy_paste,
                'translate': args.translate,
                'scale': args.scale,
                'fliplr': args.fliplr,
                'degrees': args.degrees,
                'perspective': 0.0001,  # slight perspective transform
                'hsv_h': 0.015,  # hue variation
                'hsv_s': 0.2,    # saturation variation
                'hsv_v': 0.1,    # value variation (brightness)
            }
        
        # Train model
        logger.info(f"Starting training with {args.epochs} epochs")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.img,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            weight_decay=args.weight_decay,
            cos_lr=args.cos_lr,
            save_period=args.save_period,
            exist_ok=args.exist_ok,
            **augment_params
        )
        
        # Save model
        models_dir = Path('models')
        best_model_path = models_dir / 'best_ppe_model.pt'
        last_model_path = models_dir / 'last_ppe_model.pt'
        
        # Copy the best model
        logger.info(f"Saving best model to {best_model_path}")
        shutil.copy(
            Path(args.project) / args.name / 'weights' / 'best.pt',
            best_model_path
        )
        
        # Copy the last model
        logger.info(f"Saving last model to {last_model_path}")
        shutil.copy(
            Path(args.project) / args.name / 'weights' / 'last.pt',
            last_model_path
        )
        
        logger.info(f"Training completed. Results: {results}")
        logger.info(f"Best model saved to {best_model_path}")
        logger.info(f"Last model saved to {last_model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main() 
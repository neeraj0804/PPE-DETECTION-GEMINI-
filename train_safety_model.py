#!/usr/bin/env python3
"""
Train a YOLOv8 model for safety detection using the safety.v1i.yolov8 dataset.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("safety_training_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for safety detection')
    
    parser.add_argument('--data', type=str, default='safety_data.yaml',
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
    parser.add_argument('--name', type=str, default=f'safety_exp_{int(time.time())}',
                      help='Experiment name (default: timestamped experiment name)')
    parser.add_argument('--patience', type=int, default=30,
                      help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='Adam',
                      help='Optimizer (SGD, Adam, AdamW)')
    parser.add_argument('--lr0', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                      help='Weight decay')
    parser.add_argument('--augment', action='store_true', default=True,
                      help='Use data augmentation')
    parser.add_argument('--save-period', type=int, default=10,
                      help='Save checkpoint every x epochs')
    parser.add_argument('--exist-ok', action='store_true',
                      help='Overwrite existing experiment directory')
    
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
        name = f"safety_exp_{int(time.time())}"
        exp_dir = project_dir / name
    
    exp_dir.mkdir(exist_ok=exist_ok, parents=True)
    logger.info(f"Created experiment directory: {exp_dir}")
    
    # Create models directory
    models_dir = Path('models/safety')
    models_dir.mkdir(exist_ok=True, parents=True)
    
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
    else:
        logger.warning("CUDA is not available. Training will use CPU, which may be slow.")
        args.device = 'cpu'
    
    # Print training configuration
    logger.info("\n=== Safety Dataset Training Configuration ===")
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
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Data augmentation: {'Yes' if args.augment else 'No'}")
    logger.info("==========================================\n")
    
    try:
        # Load model
        logger.info(f"Loading model from {args.weights}")
        model = YOLO(args.weights)
        
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
            weight_decay=args.weight_decay,
            augment=args.augment,
            exist_ok=True,
            save_period=args.save_period
        )
        
        # Print results
        logger.info("\n=== Training Results ===")
        logger.info(f"Best model saved at: {results.best}")
        logger.info("========================")
        
        # Copy best model to models directory
        best_model_path = Path(results.best)
        if best_model_path.exists():
            dest_path = Path('models/safety') / 'best_safety_model.pt'
            shutil.copy(best_model_path, dest_path)
            logger.info(f"Copied best model to {dest_path}")
        else:
            logger.error(f"Best model not found at {best_model_path}")
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

if __name__ == '__main__':
    main() 
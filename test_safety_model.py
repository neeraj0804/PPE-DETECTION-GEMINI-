#!/usr/bin/env python3
"""
Test a trained YOLOv8 model for safety detection on test images or videos.
"""

import argparse
import os
import logging
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("safety_testing_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description='Test trained safety detection model')
    
    parser.add_argument('--model', type=str, default='models/safety/best_safety_model.pt',
                      help='Path to the trained model')
    parser.add_argument('--data', type=str, default='safety_data.yaml',
                      help='Path to data.yaml file')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Image size for inference')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                      help='IoU threshold for NMS')
    parser.add_argument('--source', type=str, default='',
                      help='Source for inference (test folder path, image, or video)')
    parser.add_argument('--device', type=str, default='0',
                      help='Device to use (cuda device or cpu)')
    parser.add_argument('--save-dir', type=str, default='output/safety_results',
                      help='Directory to save results')
    parser.add_argument('--save-txt', action='store_true',
                      help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                      help='Save confidences in --save-txt labels')
    
    return parser.parse_args()

def process_image(model, image_path, save_dir, conf, iou, img_size, save_txt, save_conf, class_names):
    """Process a single image with the model and save results."""
    try:
        # Create output directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Process image
        logger.info(f"Processing image: {image_path}")
        
        # Run inference
        results = model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            save=True,
            save_txt=save_txt,
            save_conf=save_conf,
            project=save_dir,
            name=Path(image_path).stem
        )
        
        # Get results
        result = results[0]
        
        # Print detection summary
        boxes = result.boxes
        if len(boxes) > 0:
            logger.info(f"Detected {len(boxes)} objects:")
            for box in boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                class_name = class_names[class_id]
                logger.info(f"  {class_name}: {confidence:.3f}")
        else:
            logger.info("No objects detected")
        
        return True
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return False

def process_video(model, video_path, save_dir, conf, iou, img_size, class_names):
    """Process a video with the model and save results."""
    try:
        # Create output directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Process video
        logger.info(f"Processing video: {video_path}")
        
        # Run inference
        results = model.predict(
            source=video_path,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            save=True,
            project=save_dir,
            name=Path(video_path).stem
        )
        
        logger.info(f"Video processed and saved to {save_dir}")
        return True
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return False

def process_folder(model, folder_path, save_dir, conf, iou, img_size, save_txt, save_conf, class_names):
    """Process all images in a folder with the model."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        logger.error(f"Folder {folder_path} does not exist")
        return False
    
    # Find all images in the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    image_files = []
    video_files = []
    
    for ext in image_extensions:
        image_files.extend(list(folder_path.glob(f"*{ext}")))
        image_files.extend(list(folder_path.glob(f"*{ext.upper()}")))
    
    for ext in video_extensions:
        video_files.extend(list(folder_path.glob(f"*{ext}")))
        video_files.extend(list(folder_path.glob(f"*{ext.upper()}")))
    
    # Process images
    if image_files:
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        for image_file in image_files:
            process_image(model, str(image_file), save_dir, conf, iou, img_size, save_txt, save_conf, class_names)
    else:
        logger.warning(f"No images found in {folder_path}")
    
    # Process videos
    if video_files:
        logger.info(f"Found {len(video_files)} videos in {folder_path}")
        for video_file in video_files:
            process_video(model, str(video_file), save_dir, conf, iou, img_size, class_names)
    else:
        logger.warning(f"No videos found in {folder_path}")
    
    return True

def get_class_names(data_yaml):
    """Get class names from data.yaml file."""
    import yaml
    try:
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        class_names = data.get('names', {})
        return class_names
    except Exception as e:
        logger.error(f"Error loading class names from {data_yaml}: {e}")
        return {}

def main():
    """Main function for testing the safety model."""
    args = parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Get class names from data.yaml
    class_names = get_class_names(args.data)
    if not class_names:
        logger.error("Failed to load class names. Cannot continue.")
        return
    
    logger.info(f"Loaded {len(class_names)} class names: {class_names}")
    
    # Check CUDA availability
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        if args.device == '':
            args.device = '0'  # Default to first GPU
        logger.info(f"Using CUDA device: {args.device}")
    else:
        logger.warning("CUDA is not available. Using CPU for inference, which may be slow.")
        args.device = 'cpu'
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    try:
        model = YOLO(model_path)
        logger.info(f"Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Set source for inference
    if not args.source:
        # Use test images from dataset as default
        try:
            import yaml
            with open(args.data, 'r') as f:
                data = yaml.safe_load(f)
            
            test_path = data.get('test', data.get('val', ''))
            if test_path:
                args.source = test_path
                logger.info(f"Using test images from dataset: {args.source}")
            else:
                logger.error("No source provided and could not find test path in data.yaml")
                return
        except Exception as e:
            logger.error(f"Error loading test path from data.yaml: {e}")
            return
    
    # Process source
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source does not exist: {source_path}")
        return
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Log test settings
    logger.info("\n=== Safety Model Test Configuration ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"IoU threshold: {args.iou}")
    logger.info(f"Image size: {args.img_size}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info(f"Save text results: {args.save_txt}")
    logger.info(f"Save confidence: {args.save_conf}")
    logger.info("====================================\n")
    
    # Process source based on type
    if source_path.is_file():
        # Single file
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            # Image file
            process_image(model, str(source_path), args.save_dir, args.conf, args.iou, args.img_size, 
                        args.save_txt, args.save_conf, class_names)
        elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video file
            process_video(model, str(source_path), args.save_dir, args.conf, args.iou, args.img_size, class_names)
        else:
            logger.error(f"Unsupported file type: {source_path.suffix}")
    elif source_path.is_dir():
        # Directory
        process_folder(model, str(source_path), args.save_dir, args.conf, args.iou, args.img_size, 
                       args.save_txt, args.save_conf, class_names)
    else:
        logger.error(f"Unknown source type: {source_path}")
        return
    
    logger.info("Testing completed successfully")

if __name__ == '__main__':
    main() 
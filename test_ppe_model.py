#!/usr/bin/env python3
"""
Test a trained YOLO model for PPE detection.
"""

import os
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test YOLO model for PPE detection')
    
    parser.add_argument('--weights', type=str, default='models/best_ppe_model.pt',
                        help='Path to model weights')
    parser.add_argument('--source', type=str, default='0',
                        help='Source for detection (0 for webcam, path to image or video file)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--save', action='store_true',
                        help='Save detection results')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Check if weights file exists
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        print("Please train the model first or specify the correct path to weights.")
        return
    
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA is available with {num_gpus} GPU(s)")
        print(f"Current device: {current_device} - {device_name}")
        
        # Set device to GPU if available
        if args.device == '':
            args.device = '0'  # Default to first GPU
        
        # Check if specified device is valid
        if args.device != 'cpu' and not args.device.isdigit() and not all(d.isdigit() for d in args.device.split(',')):
            print(f"Warning: Invalid device '{args.device}'. Using default device '0'.")
            args.device = '0'
    else:
        print("CUDA is not available. Testing will use CPU, which may be slower.")
        args.device = 'cpu'
    
    # Print test configuration
    print("=== Test Configuration ===")
    print(f"Weights: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Device: {args.device}")
    print(f"Save results: {'Yes' if args.save else 'No'}")
    print("==========================")
    
    # Create output directory if saving results
    if args.save:
        os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model = YOLO(args.weights)
    
    # Class names
    class_names = {
        0: "No helmet",
        1: "No vest",
        2: "Person",
        3: "helmet",
        4: "vest"
    }
    
    # Process source
    if args.source.isdigit():
        # Webcam
        source = int(args.source)
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open webcam {source}")
            return
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Run inference
            results = model(frame, conf=args.conf, iou=args.iou, device=args.device)[0]
            
            # Process results
            annotated_frame = results.plot()
            
            # Add PPE status
            has_person = False
            has_helmet = False
            has_vest = False
            has_no_helmet = False
            has_no_vest = False
            
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id == 2:  # Person
                    has_person = True
                elif cls_id == 3:  # helmet
                    has_helmet = True
                elif cls_id == 4:  # vest
                    has_vest = True
                elif cls_id == 0:  # No helmet
                    has_no_helmet = True
                elif cls_id == 1:  # No vest
                    has_no_vest = True
            
            # Determine compliance
            if has_person:
                if (has_helmet or not has_no_helmet) and (has_vest or not has_no_vest):
                    status = "COMPLIANT"
                    color = (0, 255, 0)  # Green
                else:
                    status = "NON-COMPLIANT"
                    color = (0, 0, 255)  # Red
                
                cv2.putText(annotated_frame, status, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display the frame
            cv2.imshow("PPE Detection", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        # Image or video file
        if not os.path.exists(args.source):
            print(f"Error: Source file not found: {args.source}")
            return
        
        # Run inference
        results = model(args.source, conf=args.conf, iou=args.iou, device=args.device, save=args.save, project=args.output)
        
        # Print results
        print("\n=== Detection Results ===")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            for c in range(len(class_names)):
                count = (result.boxes.cls == c).sum().item()
                if count > 0:
                    print(f"  {class_names[c]}: {count}")
        print("========================")

if __name__ == '__main__':
    main() 
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

class PPEDetector:
    """
    Class for detecting Personal Protective Equipment (PPE) in images and video streams
    using YOLOv11 or other available YOLO models from Ultralytics.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.5, device=None):
        """
        Initialize the PPE detector.
        
        Args:
            model_path (str, optional): Path to the YOLO model weights. If None, will use a pre-trained model.
            confidence_threshold (float, optional): Confidence threshold for detections. Defaults to 0.5.
            device (str, optional): Device to use for inference ('cpu', '0', '0,1', etc.). If None, will use GPU if available.
        """
        self.confidence_threshold = confidence_threshold
        
        # Check for CUDA availability and set device
        cuda_available = torch.cuda.is_available()
        if device is None:
            if cuda_available:
                self.device = '0'  # Default to first GPU
                print(f"Using GPU for inference: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("CUDA not available. Using CPU for inference.")
        else:
            self.device = device
            if device != 'cpu' and cuda_available:
                device_id = int(device.split(',')[0]) if ',' in device else int(device)
                print(f"Using GPU for inference: {torch.cuda.get_device_name(device_id)}")
            elif device != 'cpu':
                print(f"Warning: Requested GPU device {device} but CUDA is not available. Using CPU instead.")
                self.device = 'cpu'
        
        # If no model path is provided, use the latest available YOLO model
        if model_path is None:
            # For now, we'll use YOLOv8 as YOLOv11 might not be available yet
            self.model = YOLO('yolov8n.pt')
            print("Using default YOLOv8 model. For better PPE detection, please provide a custom trained model.")
        else:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Loaded custom model from {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # PPE classes based on the downloaded dataset
        self.ppe_classes = {
            0: 'No helmet',
            1: 'No vest',
            2: 'Person',
            3: 'helmet',
            4: 'vest'
        }
        
        # Required PPE for different zones (can be customized)
        self.zone_requirements = {
            'construction': ['helmet', 'vest'],
            'chemical': ['helmet', 'vest'],
            'general': ['vest']
        }
        
    def detect_ppe(self, image, zone='construction'):
        """
        Detect PPE in an image and determine compliance.
        
        Args:
            image (numpy.ndarray): Input image (BGR format from OpenCV)
            zone (str): Work zone type to determine required PPE
            
        Returns:
            tuple: (processed_image, compliance_status, detections)
        """
        # Make a copy of the image to draw on
        output_image = image.copy()
        
        # Run inference with specified device
        results = self.model(image, conf=self.confidence_threshold, device=self.device)
        
        # Get required PPE for the zone
        required_ppe = self.zone_requirements.get(zone, self.zone_requirements['general'])
        
        # Track detected PPE
        detected_ppe = set()
        has_person = False
        has_no_helmet = False
        has_no_vest = False
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get class and confidence
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Get class name
                ppe_item = self.ppe_classes.get(cls_id, f"Class {cls_id}")
                
                # Track detections
                if cls_id == 2:  # Person
                    has_person = True
                elif cls_id == 3:  # helmet
                    detected_ppe.add('helmet')
                elif cls_id == 4:  # vest
                    detected_ppe.add('vest')
                elif cls_id == 0:  # No helmet
                    has_no_helmet = True
                elif cls_id == 1:  # No vest
                    has_no_vest = True
                
                # Draw bounding box
                if cls_id in [0, 1]:  # No helmet, No vest
                    color = (0, 0, 255)  # Red for violations
                elif cls_id == 2:  # Person
                    color = (255, 255, 0)  # Yellow for person
                else:  # helmet, vest
                    color = (0, 255, 0)  # Green for PPE
                
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{ppe_item}: {conf:.2f}"
                cv2.putText(output_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Check compliance
        is_compliant = True
        if has_person:
            for item in required_ppe:
                if item == 'helmet' and (item not in detected_ppe and has_no_helmet):
                    is_compliant = False
                elif item == 'vest' and (item not in detected_ppe and has_no_vest):
                    is_compliant = False
        
        # Add compliance status to image
        status_text = "COMPLIANT" if is_compliant else "NON-COMPLIANT"
        status_color = (0, 255, 0) if is_compliant else (0, 0, 255)
        cv2.putText(output_image, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # If non-compliant, list missing PPE
        if not is_compliant:
            missing_text = "Missing: "
            missing_items = []
            
            for item in required_ppe:
                if item == 'helmet' and (item not in detected_ppe and has_no_helmet):
                    missing_items.append('helmet')
                elif item == 'vest' and (item not in detected_ppe and has_no_vest):
                    missing_items.append('vest')
            
            missing_text += ", ".join(missing_items)
            cv2.putText(output_image, missing_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return output_image, is_compliant, detected_ppe
    
    def process_video_feed(self, video_source=0, zone='construction', display=True):
        """
        Process a video feed for PPE detection.
        
        Args:
            video_source: Camera index or video file path
            zone (str): Work zone type to determine required PPE
            display (bool): Whether to display the processed feed
            
        Returns:
            None
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream")
                break
            
            # Process the frame
            processed_frame, is_compliant, detected_ppe = self.detect_ppe(frame, zone)
            
            # Display the result
            if display:
                cv2.imshow('PPE Detection', processed_frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Release resources
        cap.release()
        if display:
            cv2.destroyAllWindows()
    
    def train_custom_model(self, data_yaml_path, epochs=100, batch_size=16, img_size=640):
        """
        Train a custom YOLO model for PPE detection.
        
        Args:
            data_yaml_path (str): Path to the YAML file containing dataset information
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            img_size (int): Image size for training
            
        Returns:
            Path to the trained model
        """
        # Create a new YOLO model instance
        model = YOLO('yolov8n.pt')  # Start with a pre-trained model
        
        # Train the model with GPU if available
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device
        )
        
        # Return the path to the best model
        return results.best 
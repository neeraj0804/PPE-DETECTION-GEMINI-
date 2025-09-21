import os
import cv2
import argparse
import time
import datetime
import torch
from pathlib import Path

from ppe_detector import PPEDetector
from camera_utils import CameraManager, VideoRecorder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PPE Detection System')
    
    parser.add_argument('--model', type=str, default='models/best_ppe_model.pt',
                        help='Path to custom YOLO model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--source', type=str, default='0',
                        help='Camera index, video file, or RTSP URL')
    parser.add_argument('--zone', type=str, default='construction',
                        choices=['construction', 'chemical', 'general'],
                        help='Work zone type to determine required PPE')
    parser.add_argument('--record', action='store_true',
                        help='Record video with detections')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for recorded videos')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Display resolution (WxH)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA is available with {num_gpus} GPU(s)")
        print(f"Current device: {current_device} - {device_name}")
    else:
        print("CUDA is not available. Using CPU for inference.")
        args.device = 'cpu'
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}. Using default 640x480.")
        resolution = (640, 480)
    
    # Create output directory if recording
    if args.record:
        os.makedirs(args.output, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output, f"ppe_detection_{timestamp}.mp4")
    
    # Check if model exists, if not use default
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Using default YOLOv8 model. For better PPE detection, please train a custom model first.")
        model_path = None
    
    # Initialize PPE detector
    print("Initializing PPE detector...")
    detector = PPEDetector(model_path=model_path, confidence_threshold=args.conf, device=args.device)
    
    # Initialize camera manager
    print("Initializing camera manager...")
    camera_manager = CameraManager()
    
    # Add camera
    source = args.source
    if source.isdigit():
        source = int(source)
    
    if not camera_manager.add_camera('main', source):
        print("Failed to add camera. Exiting.")
        return
    
    # Initialize video recorder if recording
    recorder = None
    if args.record:
        print(f"Initializing video recorder. Output: {output_path}")
        recorder = VideoRecorder(output_path, resolution=resolution)
        recorder.start()
    
    # Main loop
    print(f"Starting PPE detection in {args.zone} zone. Press 'q' to quit.")
    try:
        while True:
            # Get frame from camera
            frame = camera_manager.get_frame('main')
            
            if frame is None:
                print("No frame received. Exiting.")
                break
            
            # Resize frame for display
            display_frame = cv2.resize(frame, resolution)
            
            # Process frame for PPE detection
            start_time = time.time()
            processed_frame, is_compliant, detected_ppe = detector.detect_ppe(display_frame, zone=args.zone)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            
            # Add FPS to frame
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, processed_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Record frame if recording
            if recorder and recorder.is_recording:
                recorder.write_frame(processed_frame)
            
            # Display frame
            cv2.imshow('PPE Detection', processed_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Clean up
        print("Cleaning up...")
        if recorder and recorder.is_recording:
            recorder.stop()
        
        camera_manager.release_all()
        cv2.destroyAllWindows()
        
        print("Done")

if __name__ == '__main__':
    main() 
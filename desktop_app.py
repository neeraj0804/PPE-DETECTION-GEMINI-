import os
import sys
import cv2
import time
import numpy as np
import torch
from PIL import Image
from datetime import datetime
from pathlib import Path
import threading
import queue
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSlider, QFileDialog, 
                            QTabWidget, QGroupBox, QRadioButton, QMessageBox, QSplitter,
                            QFrame, QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

from ppe_detector import PPEDetector
from camera_utils import CameraManager, VideoRecorder
from safety_report_generator import SafetyReportGenerator

class VideoThread(QThread):
    """Thread for capturing and processing video frames."""
    update_frame = pyqtSignal(np.ndarray, bool, list, list)
    update_fps = pyqtSignal(float)
    
    def __init__(self, camera_manager, detector, zone='construction', report_generator=None):
        super().__init__()
        self.camera_manager = camera_manager
        self.detector = detector
        self.zone = zone
        self.report_generator = report_generator
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.recorder = None
        self.is_recording = False
    
    def run(self):
        """Main thread loop."""
        self.running = True
        while self.running:
            # Get frame from camera
            frame = self.camera_manager.get_frame('main')
            
            if frame is not None:
                # Process frame for PPE detection
                start_time = time.time()
                processed_frame, is_compliant, detected_ppe = self.detector.detect_ppe(frame, zone=self.zone)
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count >= 10:  # Update FPS every 10 frames
                    current_time = time.time()
                    elapsed_time = current_time - self.last_time
                    self.fps = self.frame_count / elapsed_time
                    self.update_fps.emit(self.fps)
                    self.frame_count = 0
                    self.last_time = current_time
                
                # Add FPS to frame
                cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", (10, processed_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Record frame if recording
                if self.is_recording and self.recorder is not None:
                    self.recorder.write_frame(processed_frame)
                
                # Get required PPE for the zone
                required_ppe = self.detector.zone_requirements.get(self.zone, [])
                
                # Calculate missing items
                missing_items = []
                for item in required_ppe:
                    if item not in detected_ppe:
                        missing_items.append(item)
                
                # Log detection data for report generation
                if self.report_generator:
                    self.report_generator.log_detection(
                        frame_number=self.frame_count,
                        is_compliant=is_compliant,
                        detected_ppe=list(detected_ppe),
                        missing_items=missing_items,
                        timestamp=datetime.now()
                    )
                
                # Emit signal with processed frame and detection results
                self.update_frame.emit(processed_frame, is_compliant, list(detected_ppe), missing_items)
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
    
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()
    
    def set_zone(self, zone):
        """Set the work zone type."""
        self.zone = zone
    
    def start_recording(self):
        """Start recording video."""
        # Create output directory
        os.makedirs('recordings', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join('recordings', f"ppe_detection_{timestamp}.mp4")
        
        # Initialize recorder
        self.recorder = VideoRecorder(output_path, resolution=(640, 480))
        if not self.recorder.start():
            return False, "Failed to start recording."
        
        self.is_recording = True
        return True, output_path
    
    def stop_recording(self):
        """Stop recording video."""
        if self.is_recording and self.recorder is not None:
            duration = self.recorder.stop()
            self.is_recording = False
            output_path = self.recorder.output_path
            self.recorder = None
            return True, duration, output_path
        return False, 0, None

class PPEDetectionApp(QMainWindow):
    """Main application window for PPE detection."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.camera_manager = None
        self.detector = None
        self.video_thread = None
        self.model_path = None
        self.confidence = 0.5
        self.zone = 'construction'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        
        # Initialize Safety Report Generator
        self.gemini_api_key = "AIzaSyAdMkl3-eQimVXHg2Q93vaZgBHh4pT5bFU"
        self.report_generator = SafetyReportGenerator(self.gemini_api_key)
        
        # Set up the UI
        self.init_ui()
        
        # Check for CUDA availability
        self.check_cuda()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("PPE Detection System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (settings)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)
        
        # Right panel (content)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 900])
        
        # Header
        header_label = QLabel("PPE Detection System")
        header_label.setAlignment(Qt.AlignCenter)
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header_label.setFont(header_font)
        right_layout.addWidget(header_label)
        
        description_label = QLabel(
            "This application uses computer vision and machine learning to detect whether "
            "workers are wearing proper Personal Protective Equipment (PPE) in real-time."
        )
        description_label.setWordWrap(True)
        right_layout.addWidget(description_label)
        
        # GPU info
        self.gpu_info_label = QLabel()
        self.gpu_info_label.setStyleSheet("background-color: #e6f7ff; padding: 10px; border-radius: 5px;")
        self.gpu_info_label.setWordWrap(True)
        right_layout.addWidget(self.gpu_info_label)
        
        # Tab widget for content
        tab_widget = QTabWidget()
        right_layout.addWidget(tab_widget)
        
        # Tab 1: Live Detection
        live_tab = QWidget()
        live_layout = QVBoxLayout()
        live_tab.setLayout(live_layout)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        live_layout.addWidget(self.video_label)
        
        # Status display
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(50)
        self.status_label.setFont(QFont("Arial", 14, QFont.Bold))
        live_layout.addWidget(self.status_label)
        
        # Detection info
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        live_layout.addWidget(self.info_label)
        
        # Tab 2: Image Upload
        image_tab = QWidget()
        image_layout = QVBoxLayout()
        image_tab.setLayout(image_layout)
        
        # Image upload button
        upload_image_btn = QPushButton("Upload Image")
        upload_image_btn.clicked.connect(self.upload_image)
        image_layout.addWidget(upload_image_btn)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: black;")
        image_layout.addWidget(self.image_label)
        
        # Image status display
        self.image_status_label = QLabel()
        self.image_status_label.setAlignment(Qt.AlignCenter)
        self.image_status_label.setMinimumHeight(50)
        self.image_status_label.setFont(QFont("Arial", 14, QFont.Bold))
        image_layout.addWidget(self.image_status_label)
        
        # Image detection info
        self.image_info_label = QLabel()
        self.image_info_label.setWordWrap(True)
        image_layout.addWidget(self.image_info_label)
        
        # Tab 3: Video Upload
        video_tab = QWidget()
        video_layout = QVBoxLayout()
        video_tab.setLayout(video_layout)
        
        # Video upload button
        upload_video_btn = QPushButton("Upload Video")
        upload_video_btn.clicked.connect(self.upload_video)
        video_layout.addWidget(upload_video_btn)
        
        # Video processing status
        self.video_processing_label = QLabel("No video processed yet.")
        self.video_processing_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_processing_label)
        
        # Add tabs to tab widget
        tab_widget.addTab(live_tab, "Live Detection")
        tab_widget.addTab(image_tab, "Image Upload")
        tab_widget.addTab(video_tab, "Video Upload")
        
        # Settings panel (left side)
        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)
        
        # Model selection
        self.model_label = QLabel("Using default YOLOv8 model")
        model_layout.addWidget(self.model_label)
        
        # Model upload button
        upload_model_btn = QPushButton("Upload Custom Model")
        upload_model_btn.clicked.connect(self.upload_model)
        model_layout.addWidget(upload_model_btn)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence Threshold:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(self.confidence * 100))
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        self.conf_value_label = QLabel(f"{self.confidence:.2f}")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        model_layout.addLayout(conf_layout)
        
        # Zone selection
        zone_layout = QHBoxLayout()
        zone_label = QLabel("Work Zone Type:")
        self.zone_combo = QComboBox()
        self.zone_combo.addItems(["construction", "chemical", "general"])
        self.zone_combo.setCurrentText(self.zone)
        self.zone_combo.currentTextChanged.connect(self.update_zone)
        zone_layout.addWidget(zone_label)
        zone_layout.addWidget(self.zone_combo)
        model_layout.addLayout(zone_layout)
        
        # Device selection
        if torch.cuda.is_available():
            device_layout = QHBoxLayout()
            device_label = QLabel("Computation Device:")
            self.device_combo = QComboBox()
            self.device_combo.addItems(["GPU (0)", "CPU"])
            self.device_combo.setCurrentIndex(0 if self.device == '0' else 1)
            self.device_combo.currentIndexChanged.connect(self.update_device)
            device_layout.addWidget(device_label)
            device_layout.addWidget(self.device_combo)
            model_layout.addLayout(device_layout)
        
        left_layout.addWidget(model_group)
        
        # Camera controls group
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)
        
        # Camera buttons
        camera_btn_layout = QHBoxLayout()
        self.start_camera_btn = QPushButton("Start Webcam")
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        camera_btn_layout.addWidget(self.start_camera_btn)
        camera_btn_layout.addWidget(self.record_btn)
        camera_layout.addLayout(camera_btn_layout)
        
        # Report generation buttons
        report_btn_layout = QVBoxLayout()
        self.start_session_btn = QPushButton("Start Safety Session")
        self.start_session_btn.clicked.connect(self.start_safety_session)
        self.end_session_btn = QPushButton("End Safety Session")
        self.end_session_btn.clicked.connect(self.end_safety_session)
        self.end_session_btn.setEnabled(False)
        self.generate_report_btn = QPushButton("Generate Safety Report")
        self.generate_report_btn.clicked.connect(self.generate_safety_report)
        self.generate_report_btn.setEnabled(False)
        
        report_btn_layout.addWidget(self.start_session_btn)
        report_btn_layout.addWidget(self.end_session_btn)
        report_btn_layout.addWidget(self.generate_report_btn)
        camera_layout.addLayout(report_btn_layout)
        
        left_layout.addWidget(camera_group)
        
        # About group
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout()
        about_group.setLayout(about_layout)
        
        about_text = QLabel(
            "This application uses YOLO object detection to identify PPE items such as "
            "helmets and safety vests.\n\n"
            "Different work zones have different PPE requirements. The system will "
            "check if all required PPE items for the selected zone are detected."
        )
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)
        
        left_layout.addWidget(about_group)
        
        # Add stretch to push everything up
        left_layout.addStretch()
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def check_cuda(self):
        """Check for CUDA availability and update GPU info label."""
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            num_gpus = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            gpu_info = f"CUDA is available with {num_gpus} GPU(s)\nCurrent device: {current_device} - {device_name}"
            self.gpu_info_label.setText(gpu_info)
        else:
            self.gpu_info_label.setText("CUDA is not available. Using CPU for inference.")
            self.device = 'cpu'
    
    def initialize_detector(self):
        """Initialize or update the PPE detector."""
        if self.detector is None or self.detector.confidence_threshold != self.confidence:
            try:
                self.detector = PPEDetector(
                    model_path=self.model_path,
                    confidence_threshold=self.confidence,
                    device=self.device
                )
                self.statusBar().showMessage("PPE detector initialized successfully!")
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize detector: {str(e)}")
                return False
        return True
    
    def toggle_camera(self):
        """Start or stop the webcam."""
        if self.camera_manager is None:
            # Start camera
            self.camera_manager = CameraManager()
            
            # Add webcam
            if not self.camera_manager.add_camera('main', 0):
                QMessageBox.critical(self, "Error", "Failed to start webcam. Please check your camera connection.")
                self.camera_manager = None
                return
            
            # Initialize detector
            if not self.initialize_detector():
                self.camera_manager.release_all()
                self.camera_manager = None
                return
            
            # Start video thread
            self.video_thread = VideoThread(self.camera_manager, self.detector, self.zone, self.report_generator)
            self.video_thread.update_frame.connect(self.update_frame)
            self.video_thread.update_fps.connect(self.update_fps)
            self.video_thread.start()
            
            # Update UI
            self.start_camera_btn.setText("Stop Webcam")
            self.record_btn.setEnabled(True)
            self.statusBar().showMessage("Webcam started")
        else:
            # Stop camera
            if self.video_thread is not None:
                if self.video_thread.is_recording:
                    self.toggle_recording()
                self.video_thread.stop()
                self.video_thread = None
            
            self.camera_manager.release_all()
            self.camera_manager = None
            
            # Update UI
            self.start_camera_btn.setText("Start Webcam")
            self.record_btn.setEnabled(False)
            self.record_btn.setText("Start Recording")
            self.video_label.clear()
            self.video_label.setStyleSheet("background-color: black;")
            self.status_label.clear()
            self.info_label.clear()
            self.statusBar().showMessage("Webcam stopped")
    
    def toggle_recording(self):
        """Start or stop recording."""
        if self.video_thread is None:
            return
        
        if not self.video_thread.is_recording:
            # Start recording
            success, message = self.video_thread.start_recording()
            if success:
                self.record_btn.setText("Stop Recording")
                self.statusBar().showMessage(f"Recording started: {message}")
            else:
                QMessageBox.warning(self, "Warning", message)
        else:
            # Stop recording
            success, duration, output_path = self.video_thread.stop_recording()
            if success:
                self.record_btn.setText("Start Recording")
                QMessageBox.information(self, "Recording Saved", 
                                       f"Recording saved to {output_path}\nDuration: {duration:.2f} seconds")
                self.statusBar().showMessage(f"Recording stopped. Duration: {duration:.2f}s")
            else:
                QMessageBox.warning(self, "Warning", "Failed to stop recording.")
    
    def update_frame(self, frame, is_compliant, detected_ppe, missing_items):
        """Update the video display with the processed frame."""
        # Convert frame to QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Display the image
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Update status
        if is_compliant:
            self.status_label.setText("COMPLIANT")
            self.status_label.setStyleSheet("color: #00cc00; font-weight: bold; background-color: rgba(0, 204, 0, 0.1); border-radius: 5px;")
        else:
            self.status_label.setText("NON-COMPLIANT")
            self.status_label.setStyleSheet("color: #ff0000; font-weight: bold; background-color: rgba(255, 0, 0, 0.1); border-radius: 5px;")
        
        # Update info
        info_text = f"Detected PPE: {', '.join(detected_ppe) if detected_ppe else 'None'}\n"
        if missing_items:
            info_text += f"Missing PPE: {', '.join(missing_items)}"
        self.info_label.setText(info_text)
    
    def update_fps(self, fps):
        """Update the FPS counter in the status bar."""
        self.statusBar().showMessage(f"FPS: {fps:.1f}")
    
    def update_confidence(self):
        """Update the confidence threshold."""
        self.confidence = self.conf_slider.value() / 100.0
        self.conf_value_label.setText(f"{self.confidence:.2f}")
        
        # Update detector if running
        if self.detector is not None:
            self.detector.confidence_threshold = self.confidence
    
    def update_zone(self, zone):
        """Update the work zone type."""
        self.zone = zone
        
        # Update video thread if running
        if self.video_thread is not None:
            self.video_thread.set_zone(zone)
    
    def update_device(self, index):
        """Update the computation device."""
        self.device = '0' if index == 0 else 'cpu'
        
        # Reset detector to use new device
        if self.detector is not None:
            self.detector = None
            if self.video_thread is not None:
                # Restart video thread
                self.toggle_camera()  # Stop
                self.toggle_camera()  # Start
    
    def upload_model(self):
        """Upload a custom YOLO model."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Models (*.pt)")
        if file_path:
            self.model_path = file_path
            self.model_label.setText(f"Using model: {os.path.basename(file_path)}")
            
            # Reset detector to use new model
            self.detector = None
            
            # Restart video thread if running
            if self.video_thread is not None:
                self.toggle_camera()  # Stop
                self.toggle_camera()  # Start
    
    def upload_image(self):
        """Upload and process an image."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            try:
                # Initialize detector if needed
                if not self.initialize_detector():
                    return
                
                # Read image
                image = cv2.imread(file_path)
                if image is None:
                    QMessageBox.critical(self, "Error", "Failed to read image file.")
                    return
                
                # Process image
                processed_image, is_compliant, detected_ppe = self.detector.detect_ppe(image, zone=self.zone)
                
                # Convert to RGB for display
                rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                # Convert to QImage and display
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.image_label.width(), self.image_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                
                # Update status
                if is_compliant:
                    self.image_status_label.setText("COMPLIANT")
                    self.image_status_label.setStyleSheet("color: #00cc00; font-weight: bold; background-color: rgba(0, 204, 0, 0.1); border-radius: 5px;")
                else:
                    self.image_status_label.setText("NON-COMPLIANT")
                    self.image_status_label.setStyleSheet("color: #ff0000; font-weight: bold; background-color: rgba(255, 0, 0, 0.1); border-radius: 5px;")
                
                # Get required PPE for the zone
                required_ppe = self.detector.zone_requirements.get(self.zone, [])
                
                # Calculate missing items
                missing_items = []
                for item in required_ppe:
                    if item not in detected_ppe:
                        missing_items.append(item)
                
                # Update info
                info_text = f"Detected PPE: {', '.join(detected_ppe) if detected_ppe else 'None'}\n"
                if missing_items:
                    info_text += f"Missing PPE: {', '.join(missing_items)}"
                self.image_info_label.setText(info_text)
                
                self.statusBar().showMessage(f"Processed image: {os.path.basename(file_path)}")
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error processing image: {str(e)}")
    
    def upload_video(self):
        """Upload and process a video."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            try:
                # Initialize detector if needed
                if not self.initialize_detector():
                    return
                
                # Create output directory
                os.makedirs('processed_videos', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join('processed_videos', f"processed_{timestamp}.mp4")
                
                # Show processing message
                self.video_processing_label.setText("Processing video... Please wait.")
                QApplication.processEvents()
                
                # Process video in a separate thread to avoid freezing the UI
                self.process_video_thread = threading.Thread(
                    target=self.process_video,
                    args=(file_path, output_path)
                )
                self.process_video_thread.daemon = True
                self.process_video_thread.start()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error processing video: {str(e)}")
                self.video_processing_label.setText("Error processing video.")
    
    def process_video(self, input_path, output_path):
        """Process a video file."""
        try:
            # Open video file
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.video_processing_label.setText("Error: Could not open video file.")
                return
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, _, _ = self.detector.detect_ppe(frame, zone=self.zone)
                
                # Write frame
                out.write(processed_frame)
                
                # Update progress
                frame_count += 1
                progress = int(frame_count / total_frames * 100)
                self.video_processing_label.setText(f"Processing video: {progress}% complete")
                QApplication.processEvents()
            
            # Release resources
            cap.release()
            out.release()
            
            # Show completion message
            self.video_processing_label.setText(f"Video processed successfully!\nSaved to: {output_path}")
            
        except Exception as e:
            self.video_processing_label.setText(f"Error processing video: {str(e)}")
    
    def start_safety_session(self):
        """Start a new safety monitoring session."""
        try:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.report_generator.start_session(
                session_id=session_id,
                zone=self.zone,
                model_used=os.path.basename(self.model_path) if self.model_path else "default_yolov8"
            )
            
            self.start_session_btn.setEnabled(False)
            self.end_session_btn.setEnabled(True)
            self.statusBar().showMessage(f"Safety session started: {session_id}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start safety session: {str(e)}")
    
    def end_safety_session(self):
        """End the current safety monitoring session."""
        try:
            self.report_generator.end_session()
            
            self.start_session_btn.setEnabled(True)
            self.end_session_btn.setEnabled(False)
            self.generate_report_btn.setEnabled(True)
            self.statusBar().showMessage("Safety session ended. Ready to generate report.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to end safety session: {str(e)}")
    
    def generate_safety_report(self):
        """Generate a comprehensive safety report."""
        try:
            if not self.report_generator.detection_data:
                QMessageBox.warning(self, "Warning", "No detection data available. Please run a safety session first.")
                return
            
            # Show progress dialog
            progress_dialog = QMessageBox(self)
            progress_dialog.setWindowTitle("Generating Report")
            progress_dialog.setText("Generating safety report... This may take a few moments.")
            progress_dialog.setStandardButtons(QMessageBox.NoButton)
            progress_dialog.show()
            QApplication.processEvents()
            
            # Generate the complete report
            report_files = self.report_generator.generate_complete_report()
            
            progress_dialog.close()
            
            if "error" in report_files:
                QMessageBox.critical(self, "Error", report_files["error"])
                return
            
            # Show success message with file paths
            message = f"Safety report generated successfully!\n\n"
            message += f"PDF Report: {report_files['pdf_report']}\n"
            message += f"Excel Report: {report_files['excel_report']}\n"
            message += f"Compliance Chart: {report_files['compliance_chart']}\n"
            message += f"Timeline Chart: {report_files['timeline_chart']}\n\n"
            message += f"Compliance Rate: {report_files['summary']['compliance_rate']:.1f}%"
            
            QMessageBox.information(self, "Report Generated", message)
            
            # Ask if user wants to open the reports folder
            reply = QMessageBox.question(self, "Open Reports Folder", 
                                       "Would you like to open the reports folder?",
                                       QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                import subprocess
                import platform
                
                reports_folder = os.path.abspath("reports")
                if platform.system() == "Windows":
                    subprocess.run(["explorer", reports_folder])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", reports_folder])
                else:  # Linux
                    subprocess.run(["xdg-open", reports_folder])
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate safety report: {str(e)}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop video thread if running
        if self.video_thread is not None:
            self.video_thread.stop()
        
        # Release camera resources
        if self.camera_manager is not None:
            self.camera_manager.release_all()
        
        event.accept()

def main():
    """Main function."""
    app = QApplication(sys.argv)
    window = PPEDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 
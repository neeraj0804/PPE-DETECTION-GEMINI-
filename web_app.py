import os
import cv2
import time
import numpy as np
import streamlit as st
import torch
from PIL import Image
import tempfile
from datetime import datetime
from pathlib import Path
import threading
import queue
import base64

from ppe_detector import PPEDetector
from camera_utils import CameraManager, VideoRecorder
from safety_report_generator import SafetyReportGenerator

# Set page configuration
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0078ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0078ff;
        margin-bottom: 1rem;
    }
    .status-compliant {
        font-size: 1.8rem;
        color: #00cc00;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(0, 204, 0, 0.1);
        text-align: center;
    }
    .status-non-compliant {
        font-size: 1.8rem;
        color: #ff0000;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 0, 0, 0.1);
        text-align: center;
    }
    .info-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    .gpu-info {
        padding: 10px;
        border-radius: 5px;
        background-color: #e6f7ff;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .stVideo {
        width: 100%;
        height: auto;
    }
    .video-container {
        position: relative;
        width: 100%;
        padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
        height: 0;
        overflow: hidden;
    }
    .video-container img {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .fps-counter {
        position: absolute;
        bottom: 10px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        padding: 5px;
        border-radius: 3px;
        font-size: 0.8rem;
    }
    .stApp {
        margin: 0;
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)

# Check for CUDA availability
cuda_available = torch.cuda.is_available()
if cuda_available:
    num_gpus = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    gpu_info = f"Using GPU: {device_name}"
    device = '0'  # Default to first GPU
else:
    gpu_info = "CUDA not available. Using CPU for inference."
    device = 'cpu'

# Initialize session state
if 'camera_manager' not in st.session_state:
    st.session_state.camera_manager = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'is_webcam_active' not in st.session_state:
    st.session_state.is_webcam_active = False
if 'recorder' not in st.session_state:
    st.session_state.recorder = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'zone' not in st.session_state:
    st.session_state.zone = 'construction'
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.5
if 'model_path' not in st.session_state:
    # Check if trained model exists
    default_model_path = 'models/best_ppe_model.pt'
    if os.path.exists(default_model_path):
        st.session_state.model_path = default_model_path
    else:
        st.session_state.model_path = None
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = time.time()
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'device' not in st.session_state:
    st.session_state.device = device
if 'processed_frame' not in st.session_state:
    st.session_state.processed_frame = None
if 'is_compliant' not in st.session_state:
    st.session_state.is_compliant = None
if 'detected_ppe' not in st.session_state:
    st.session_state.detected_ppe = set()
if 'missing_items' not in st.session_state:
    st.session_state.missing_items = []
if 'placeholder' not in st.session_state:
    st.session_state.placeholder = None
if 'report_generator' not in st.session_state:
    gemini_api_key = "AIzaSyAdMkl3-eQimVXHg2Q93vaZgBHh4pT5bFU"
    st.session_state.report_generator = SafetyReportGenerator(gemini_api_key)
if 'safety_session_active' not in st.session_state:
    st.session_state.safety_session_active = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = None

def initialize_detector():
    """Initialize or update the PPE detector."""
    if st.session_state.detector is None or st.session_state.confidence != st.session_state.detector.confidence_threshold:
        st.session_state.detector = PPEDetector(
            model_path=st.session_state.model_path,
            confidence_threshold=st.session_state.confidence,
            device=st.session_state.device
        )
        st.success("PPE detector initialized successfully!")

def start_webcam():
    """Start the webcam."""
    if st.session_state.camera_manager is None:
        st.session_state.camera_manager = CameraManager()
    
    # Add webcam
    if not st.session_state.camera_manager.add_camera('webcam', 0):
        st.error("Failed to start webcam. Please check your camera connection.")
        return False
    
    st.session_state.is_webcam_active = True
    return True

def stop_webcam():
    """Stop the webcam."""
    if st.session_state.camera_manager is not None:
        st.session_state.camera_manager.release_all()
        st.session_state.camera_manager = None
    
    st.session_state.is_webcam_active = False
    
    # Stop recording if active
    if st.session_state.is_recording and st.session_state.recorder is not None:
        st.session_state.recorder.stop()
        st.session_state.is_recording = False
        st.session_state.recorder = None

def process_frame(frame, zone='construction'):
    """Process a single frame for PPE detection."""
    # Initialize detector if needed
    if st.session_state.detector is None:
        initialize_detector()
    
    # Process frame
    start_time = time.time()
    processed_frame, is_compliant, detected_ppe = st.session_state.detector.detect_ppe(
        frame, zone=zone
    )
    
    # Update FPS calculation
    st.session_state.frame_count += 1
    if st.session_state.frame_count >= 10:  # Update FPS every 10 frames
        elapsed_time = time.time() - st.session_state.last_detection_time
        st.session_state.fps = st.session_state.frame_count / elapsed_time
        st.session_state.frame_count = 0
        st.session_state.last_detection_time = time.time()
    
    # Add FPS to frame
    cv2.putText(
        processed_frame,
        f"FPS: {st.session_state.fps:.1f}",
        (10, processed_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )
    
    # Record frame if recording
    if st.session_state.is_recording and st.session_state.recorder is not None:
        st.session_state.recorder.write_frame(processed_frame)
    
    # Convert to RGB for display
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # Calculate missing items
    required_ppe = st.session_state.detector.zone_requirements.get(
        zone, []
    )
    missing_items = []
    for item in required_ppe:
        if item not in detected_ppe:
            missing_items.append(item)
    
    # Log detection data for report generation if session is active
    if st.session_state.safety_session_active and st.session_state.report_generator:
        st.session_state.report_generator.log_detection(
            frame_number=st.session_state.frame_count,
            is_compliant=is_compliant,
            detected_ppe=list(detected_ppe),
            missing_items=missing_items,
            timestamp=datetime.now()
        )
    
    return processed_frame_rgb, is_compliant, detected_ppe, missing_items

def get_image_as_base64(image):
    """Convert an image to base64 string for HTML display."""
    buffered = tempfile.NamedTemporaryFile(suffix=".jpg")
    pil_img = Image.fromarray(image)
    pil_img.save(buffered.name)
    with open(buffered.name, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode()
    return img_str

def start_recording():
    """Start recording video."""
    if not st.session_state.is_webcam_active:
        st.error("Webcam is not active. Please start the webcam first.")
        return False
    
    # Create output directory
    os.makedirs('recordings', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join('recordings', f"ppe_detection_{timestamp}.mp4")
    
    # Initialize recorder
    st.session_state.recorder = VideoRecorder(output_path, resolution=(640, 480))
    if not st.session_state.recorder.start():
        st.error("Failed to start recording.")
        st.session_state.recorder = None
        return False
    
    st.session_state.is_recording = True
    return True

def stop_recording():
    """Stop recording video."""
    if st.session_state.is_recording and st.session_state.recorder is not None:
        duration = st.session_state.recorder.stop()
        st.session_state.is_recording = False
        st.session_state.recorder = None
        return duration
    return 0

def process_uploaded_image(uploaded_file):
    """Process an uploaded image for PPE detection."""
    if uploaded_file is None:
        return None, None, None, None
    
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process image
    return process_frame(image, zone=st.session_state.zone)

def process_uploaded_video(uploaded_file):
    """Process an uploaded video for PPE detection."""
    if uploaded_file is None:
        return None
    
    # Save uploaded file to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Initialize detector if needed
    initialize_detector()
    
    # Process video
    cap = cv2.VideoCapture(tfile.name)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        os.unlink(tfile.name)
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('processed_videos', exist_ok=True)
    output_path = os.path.join('processed_videos', f"processed_{timestamp}.mp4")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frames
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, is_compliant, detected_ppe = st.session_state.detector.detect_ppe(
            frame, zone=st.session_state.zone
        )
        
        # Write frame to output video
        out.write(processed_frame)
        
        # Update progress
        frame_count += 1
        progress = int(frame_count / total_frames * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {progress}% ({frame_count}/{total_frames} frames)")
    
    # Release resources
    cap.release()
    out.release()
    os.unlink(tfile.name)
    
    # Return path to processed video
    return output_path

def main():
    """Main function for the Streamlit app."""
    # Header
    st.markdown("<h1 class='main-header'>PPE Detection System</h1>", unsafe_allow_html=True)
    st.markdown(
        "This application uses computer vision and machine learning to detect whether "
        "workers are wearing proper Personal Protective Equipment (PPE) in real-time."
    )
    
    # Display GPU info
    st.markdown(f"<div class='gpu-info'>{gpu_info}</div>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
        
        # Model settings
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("<h3>Model Settings</h3>", unsafe_allow_html=True)
        
        # Display current model status
        if st.session_state.model_path:
            st.success(f"Using model: {os.path.basename(st.session_state.model_path)}")
        else:
            st.warning("Using default YOLOv8 model. For better PPE detection, please upload a custom model.")
        
        uploaded_model = st.file_uploader("Upload custom YOLO model (optional)", type=['pt'])
        if uploaded_model is not None:
            # Save model to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(uploaded_model.getbuffer())
                st.session_state.model_path = tmp_file.name
                st.success(f"Model uploaded: {uploaded_model.name}")
                # Reset detector to use new model
                st.session_state.detector = None
        
        st.session_state.confidence = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.confidence,
            step=0.05
        )
        
        st.session_state.zone = st.selectbox(
            "Work Zone Type",
            options=["construction", "chemical", "general"],
            index=["construction", "chemical", "general"].index(st.session_state.zone)
        )
        
        # Device selection
        if cuda_available:
            device_options = ['0', 'cpu']
            device_index = 0 if st.session_state.device == '0' else 1
            st.session_state.device = st.selectbox(
                "Computation Device",
                options=device_options,
                index=device_index,
                help="Select GPU (0) for faster processing or CPU for compatibility"
            )
            if st.session_state.device != device:
                # Reset detector to use new device
                st.session_state.detector = None
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Camera controls
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("<h3>Camera Controls</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.is_webcam_active:
                if st.button("Start Webcam"):
                    start_webcam()
            else:
                if st.button("Stop Webcam"):
                    stop_webcam()
        
        with col2:
            if not st.session_state.is_recording:
                if st.button("Start Recording", disabled=not st.session_state.is_webcam_active):
                    start_recording()
            else:
                if st.button("Stop Recording"):
                    duration = stop_recording()
                    st.success(f"Recording saved! Duration: {duration:.2f}s")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Safety Session Controls
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("<h3>Safety Session Controls</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.safety_session_active:
                if st.button("Start Safety Session", type="primary"):
                    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.report_generator.start_session(
                        session_id=session_id,
                        zone=st.session_state.zone,
                        model_used=os.path.basename(st.session_state.model_path) if st.session_state.model_path else "default_yolov8"
                    )
                    st.session_state.safety_session_active = True
                    st.session_state.session_id = session_id
                    st.success(f"Safety session started: {session_id}")
                    st.experimental_rerun()
            else:
                if st.button("End Safety Session", type="secondary"):
                    st.session_state.report_generator.end_session()
                    st.session_state.safety_session_active = False
                    st.success("Safety session ended. Ready to generate report.")
                    st.experimental_rerun()
        
        with col2:
            if st.session_state.safety_session_active:
                st.info(f"Session: {st.session_state.session_id}")
            else:
                st.info("No active session")
        
        # Generate Report Button
        if st.button("Generate Safety Report", disabled=not st.session_state.safety_session_active):
            if not st.session_state.report_generator.detection_data:
                st.warning("No detection data available. Please run a safety session first.")
            else:
                with st.spinner("Generating safety report... This may take a few moments."):
                    try:
                        report_files = st.session_state.report_generator.generate_complete_report()
                        
                        if "error" in report_files:
                            st.error(report_files["error"])
                        else:
                            st.success("Safety report generated successfully!")
                            
                            # Display summary
                            summary = report_files['summary']
                            st.markdown(f"**Compliance Rate:** {summary['compliance_rate']:.1f}%")
                            st.markdown(f"**Total Frames:** {summary['total_frames']}")
                            st.markdown(f"**Compliant Frames:** {summary['compliant_frames']}")
                            st.markdown(f"**Non-Compliant Frames:** {summary['non_compliant_frames']}")
                            
                            # Display AI insights with markdown formatting
                            st.markdown("**AI-Powered Safety Analysis:**")
                            
                            # Parse and display AI insights with proper markdown
                            ai_insights = report_files['ai_insights']
                            
                            # Split into sections and display with proper formatting
                            sections = ai_insights.split('##')
                            for i, section in enumerate(sections):
                                if i == 0:  # First section (before any ##)
                                    if section.strip():
                                        st.markdown(section.strip())
                                else:
                                    # Extract section title and content
                                    lines = section.strip().split('\n')
                                    if lines:
                                        title = lines[0].strip()
                                        content = '\n'.join(lines[1:]).strip()
                                        
                                        # Display section with proper formatting
                                        if title:
                                            st.markdown(f"### {title}")
                                        if content:
                                            # Convert bullet points to proper markdown
                                            content_lines = content.split('\n')
                                            for line in content_lines:
                                                line = line.strip()
                                                if line.startswith('-'):
                                                    st.markdown(f"• {line[1:].strip()}")
                                                elif line:
                                                    st.markdown(line)
                                        
                                        st.markdown("---")  # Separator between sections
                            
                            # Provide download links
                            st.markdown("**Download Reports:**")
                            
                            # PDF Report
                            if os.path.exists(report_files['pdf_report']):
                                with open(report_files['pdf_report'], 'rb') as file:
                                    st.download_button(
                                        label="Download PDF Report",
                                        data=file,
                                        file_name=os.path.basename(report_files['pdf_report']),
                                        mime="application/pdf"
                                    )
                            
                            # Excel Report
                            if os.path.exists(report_files['excel_report']):
                                with open(report_files['excel_report'], 'rb') as file:
                                    st.download_button(
                                        label="Download Excel Report",
                                        data=file,
                                        file_name=os.path.basename(report_files['excel_report']),
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            
                            # Charts
                            if os.path.exists(report_files['compliance_chart']):
                                st.image(report_files['compliance_chart'], caption="Compliance Chart")
                            
                            if os.path.exists(report_files['timeline_chart']):
                                st.image(report_files['timeline_chart'], caption="Timeline Chart")
                    
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # About
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("<h3>About</h3>", unsafe_allow_html=True)
        st.markdown(
            "This application uses YOLO object detection to identify PPE items such as "
            "helmets and safety vests."
        )
        st.markdown(
            "Different work zones have different PPE requirements. The system will "
            "check if all required PPE items for the selected zone are detected."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Live Detection", "Image Upload", "Video Upload"])
    
    # Tab 1: Live Detection
    with tab1:
        if not st.session_state.is_webcam_active:
            st.info("Click 'Start Webcam' in the sidebar to begin live detection.")
        else:
            # Initialize detector if needed
            initialize_detector()
            
            # Create placeholders for the video feed and status
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            info_placeholder = st.empty()
            
            # Get frame from webcam
            frame = st.session_state.camera_manager.get_frame('webcam')
            
            if frame is not None:
                # Process the frame
                processed_frame, is_compliant, detected_ppe, missing_items = process_frame(
                    frame, zone=st.session_state.zone
                )
                
                # Display the processed frame
                video_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                
                # Display status
                if is_compliant:
                    status_placeholder.markdown(
                        "<div class='status-compliant'>✅ COMPLIANT</div>",
                        unsafe_allow_html=True
                    )
                else:
                    status_placeholder.markdown(
                        f"<div class='status-non-compliant'>❌ NON-COMPLIANT<br>"
                        f"<span style='font-size: 1rem;'>Missing: {', '.join(missing_items)}</span></div>",
                        unsafe_allow_html=True
                    )
                
                # Display detection info
                info_placeholder.markdown(
                    f"<div class='info-box'>"
                    f"<h3>Detection Info</h3>"
                    f"<p>Zone: {st.session_state.zone}</p>"
                    f"<p>Detected PPE: {', '.join(detected_ppe) if detected_ppe else 'None'}</p>"
                    f"<p>Required PPE: {', '.join(st.session_state.detector.zone_requirements.get(st.session_state.zone, []))}</p>"
                    f"<p>FPS: {st.session_state.fps:.1f}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            # Add a refresh button
            if st.button("Refresh Feed"):
                st.experimental_rerun()
            
            # Add auto-refresh message
            st.info("Click 'Refresh Feed' to update the camera view. For continuous updates, refresh every few seconds.")
    
    # Tab 2: Image Upload
    with tab2:
        uploaded_image = st.file_uploader("Upload an image for PPE detection", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            # Process image
            processed_image, is_compliant, detected_ppe, missing_items = process_uploaded_image(uploaded_image)
            
            if processed_image is not None:
                # Display image
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                
                # Display status
                if is_compliant:
                    st.markdown(
                        "<div class='status-compliant'>✅ COMPLIANT</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='status-non-compliant'>❌ NON-COMPLIANT<br>"
                        f"<span style='font-size: 1rem;'>Missing: {', '.join(missing_items)}</span></div>",
                        unsafe_allow_html=True
                    )
                
                # Display detection info
                st.markdown(
                    f"<div class='info-box'>"
                    f"<h3>Detection Info</h3>"
                    f"<p>Zone: {st.session_state.zone}</p>"
                    f"<p>Detected PPE: {', '.join(detected_ppe) if detected_ppe else 'None'}</p>"
                    f"<p>Required PPE: {', '.join(st.session_state.detector.zone_requirements.get(st.session_state.zone, []))}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
    # Tab 3: Video Upload
    with tab3:
        uploaded_video = st.file_uploader("Upload a video for PPE detection", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    output_path = process_uploaded_video(uploaded_video)
                
                if output_path is not None and os.path.exists(output_path):
                    st.success("Video processing complete!")
                    
                    # Display processed video
                    st.video(output_path)
                    
                    # Provide download link
                    with open(output_path, 'rb') as file:
                        st.download_button(
                            label="Download Processed Video",
                            data=file,
                            file_name=os.path.basename(output_path),
                            mime="video/mp4"
                        )

if __name__ == "__main__":
    main() 
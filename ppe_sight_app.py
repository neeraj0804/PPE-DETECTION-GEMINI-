#!/usr/bin/env python3
"""
PPE Sight Dashboard - Integrated Web Application
Combines the frontend design with the AI-powered Safety Report Generator backend.
"""

import os
import sys
import cv2
import time
import json
import base64
import numpy as np
import streamlit as st
import torch
from PIL import Image
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppe_detector import PPEDetector
from camera_utils import CameraManager, VideoRecorder
from safety_report_generator import SafetyReportGenerator

# Set page configuration
st.set_page_config(
    page_title="PPE Sight Dashboard",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dashboard look
st.markdown("""
<style>
    /* Main Dashboard Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status Indicators */
    .status-excellent {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-good {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-critical {
        background: linear-gradient(135deg, #8e44ad, #9b59b6);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Alert Styling */
    .alert-success {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Video Container */
    .video-container {
        background: #000;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Report Cards */
    .report-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .report-card h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
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
        st.session_state.device = '0' if torch.cuda.is_available() else 'cpu'
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
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = {
            'total_sessions': 0,
            'total_violations': 0,
            'compliance_rate': 0,
            'active_workers': 0,
            'recent_reports': []
        }

def get_status_class(compliance_rate):
    """Get CSS class based on compliance rate."""
    if compliance_rate >= 90:
        return "status-excellent"
    elif compliance_rate >= 80:
        return "status-good"
    elif compliance_rate >= 70:
        return "status-warning"
    else:
        return "status-critical"

def get_status_text(compliance_rate):
    """Get status text based on compliance rate."""
    if compliance_rate >= 90:
        return "EXCELLENT"
    elif compliance_rate >= 80:
        return "GOOD"
    elif compliance_rate >= 70:
        return "NEEDS IMPROVEMENT"
    else:
        return "CRITICAL"

def create_metric_card(title, value, subtitle="", icon="📊"):
    """Create a metric card component."""
    return f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <h3 style="margin: 0; color: #2c3e50;">{title}</h3>
        </div>
        <p class="metric-value">{value}</p>
        <p class="metric-label">{subtitle}</p>
    </div>
    """

def create_compliance_chart(data):
    """Create a compliance chart using Plotly."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(data.keys()),
            y=list(data.values()),
            marker_color=['#2ecc71' if v >= 80 else '#f39c12' if v >= 70 else '#e74c3c' for v in data.values()],
            text=[f"{v}%" for v in data.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="PPE Compliance by Zone",
        xaxis_title="Work Zones",
        yaxis_title="Compliance Rate (%)",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_violations_timeline(data):
    """Create a violations timeline chart."""
    if not data:
        return None
    
    # Sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    violations = np.random.poisson(5, len(dates))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=violations,
        mode='lines+markers',
        name='Violations',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Daily Violations Trend",
        xaxis_title="Date",
        yaxis_title="Number of Violations",
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🦺 PPE Sight Dashboard</h1>
        <p>AI-Powered Safety Monitoring & Compliance Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        
        # System Status
        st.markdown("### 📊 System Status")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            st.success(f"🟢 GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("🟡 Using CPU")
        
        # Settings
        st.markdown("### 🔧 Settings")
        
        # Model selection
        if st.session_state.model_path:
            st.success(f"✅ Model: {os.path.basename(st.session_state.model_path)}")
        else:
            st.warning("⚠️ Using default model")
        
        # Confidence threshold
        st.session_state.confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.confidence,
            step=0.05
        )
        
        # Work zone
        st.session_state.zone = st.selectbox(
            "Work Zone",
            options=["construction", "chemical", "general"],
            index=["construction", "chemical", "general"].index(st.session_state.zone)
        )
        
        # Device selection
        if cuda_available:
            device_options = ['0', 'cpu']
            device_index = 0 if st.session_state.device == '0' else 1
            st.session_state.device = st.selectbox(
                "Processing Device",
                options=device_options,
                index=device_index
            )
        
        st.markdown("---")
        
        # Camera Controls
        st.markdown("### 📹 Camera Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.is_webcam_active:
                if st.button("🎥 Start Camera", use_container_width=True):
                    # Initialize camera
                    st.session_state.camera_manager = CameraManager()
                    if st.session_state.camera_manager.add_camera('webcam', 0):
                        st.session_state.is_webcam_active = True
                        st.success("Camera started!")
                        st.rerun()
                    else:
                        st.error("Failed to start camera")
            else:
                if st.button("⏹️ Stop Camera", use_container_width=True):
                    if st.session_state.camera_manager:
                        st.session_state.camera_manager.release_all()
                        st.session_state.camera_manager = None
                    st.session_state.is_webcam_active = False
                    st.success("Camera stopped!")
                    st.rerun()
        
        with col2:
            if st.session_state.is_webcam_active:
                if not st.session_state.is_recording:
                    if st.button("🔴 Record", use_container_width=True):
                        # Start recording logic
                        st.session_state.is_recording = True
                        st.success("Recording started!")
                else:
                    if st.button("⏹️ Stop Record", use_container_width=True):
                        st.session_state.is_recording = False
                        st.success("Recording stopped!")
        
        st.markdown("---")
        
        # Safety Session Controls
        st.markdown("### 🛡️ Safety Session")
        
        if not st.session_state.safety_session_active:
            if st.button("🚀 Start Session", use_container_width=True, type="primary"):
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.report_generator.start_session(
                    session_id=session_id,
                    zone=st.session_state.zone,
                    model_used=os.path.basename(st.session_state.model_path) if st.session_state.model_path else "default_yolov8"
                )
                st.session_state.safety_session_active = True
                st.session_state.session_id = session_id
                st.success(f"Session started: {session_id}")
                st.rerun()
        else:
            if st.button("⏹️ End Session", use_container_width=True):
                st.session_state.report_generator.end_session()
                st.session_state.safety_session_active = False
                st.success("Session ended!")
                st.rerun()
        
        if st.session_state.safety_session_active:
            st.info(f"🟢 Active: {st.session_state.session_id}")
    
    # Main Dashboard Content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📹 Live Monitoring", "📈 Analytics", "📋 Reports"])
    
    with tab1:
        # Dashboard Overview
        st.markdown("## 📊 Safety Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            compliance_rate = 87.3  # This would come from actual data
            status_class = get_status_class(compliance_rate)
            status_text = get_status_text(compliance_rate)
            st.markdown(create_metric_card(
                "Compliance Rate",
                f"{compliance_rate}%",
                status_text,
                "🎯"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card(
                "Active Workers",
                "24",
                "Currently Monitored",
                "👥"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                "Violations Today",
                "3",
                "Requires Attention",
                "⚠️"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_metric_card(
                "Sessions Today",
                "8",
                "Safety Sessions",
                "📅"
            ), unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### 🎯 Compliance by Zone")
            
            # Sample compliance data
            compliance_data = {
                'Construction': 87,
                'Chemical': 92,
                'General': 78
            }
            
            fig = create_compliance_chart(compliance_data)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### 📈 Violations Trend")
            
            fig = create_violations_timeline({})
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No violation data available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent Activity
        st.markdown("## 📋 Recent Activity")
        
        # Sample recent activity data
        recent_activities = [
            {"time": "10:30 AM", "event": "Safety violation detected", "worker": "John Doe", "type": "Missing Helmet", "status": "Resolved"},
            {"time": "10:15 AM", "event": "Safety session completed", "worker": "Team A", "type": "Construction Zone", "status": "87% Compliance"},
            {"time": "09:45 AM", "event": "New safety report generated", "worker": "System", "type": "Daily Report", "status": "Available"},
            {"time": "09:30 AM", "event": "Safety violation detected", "worker": "Jane Smith", "type": "Missing Vest", "status": "Under Review"},
        ]
        
        for activity in recent_activities:
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                with col1:
                    st.text(activity["time"])
                with col2:
                    st.text(activity["event"])
                with col3:
                    st.text(f"{activity['worker']} - {activity['type']}")
                with col4:
                    if "Resolved" in activity["status"]:
                        st.markdown('<span class="status-excellent">Resolved</span>', unsafe_allow_html=True)
                    elif "Compliance" in activity["status"]:
                        st.markdown('<span class="status-good">Good</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-warning">Pending</span>', unsafe_allow_html=True)
                st.markdown("---")
    
    with tab2:
        # Live Monitoring
        st.markdown("## 📹 Live PPE Monitoring")
        
        if not st.session_state.is_webcam_active:
            st.markdown("""
            <div class="alert-warning">
                <h3>⚠️ Camera Not Active</h3>
                <p>Please start the camera from the sidebar to begin live monitoring.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Initialize detector if needed
            if st.session_state.detector is None:
                st.session_state.detector = PPEDetector(
                    model_path=st.session_state.model_path,
                    confidence_threshold=st.session_state.confidence,
                    device=st.session_state.device
                )
            
            # Get frame from camera
            frame = st.session_state.camera_manager.get_frame('webcam')
            
            if frame is not None:
                # Process frame
                processed_frame, is_compliant, detected_ppe = st.session_state.detector.detect_ppe(
                    frame, zone=st.session_state.zone
                )
                
                # Log detection data for report generation if session is active
                if st.session_state.safety_session_active and st.session_state.report_generator:
                    st.session_state.report_generator.log_detection(
                        frame_number=st.session_state.frame_count,
                        is_compliant=is_compliant,
                        detected_ppe=list(detected_ppe),
                        missing_items=[],
                        timestamp=datetime.now()
                    )
                    st.session_state.frame_count += 1
                
                # Convert to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display the processed frame
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.image(processed_frame_rgb, caption="Live PPE Detection", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Status display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if is_compliant:
                        st.markdown('<div class="alert-success"><h3>✅ COMPLIANT</h3><p>All required PPE detected</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-danger"><h3>❌ NON-COMPLIANT</h3><p>Missing required PPE</p></div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Detected PPE", ", ".join(detected_ppe) if detected_ppe else "None")
                
                with col3:
                    st.metric("FPS", f"{st.session_state.fps:.1f}")
                
                # Refresh button
                if st.button("🔄 Refresh Feed"):
                    st.rerun()
            else:
                st.error("Failed to capture frame from camera")
    
    with tab3:
        # Analytics
        st.markdown("## 📈 Safety Analytics")
        
        # Analytics content would go here
        st.info("Analytics features coming soon...")
    
    with tab4:
        # Reports
        st.markdown("## 📋 Safety Reports")
        
        if st.session_state.safety_session_active:
            st.markdown("""
            <div class="alert-success">
                <h3>🟢 Session Active</h3>
                <p>Safety session is currently running. Data is being collected for report generation.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📊 Generate Report", type="primary"):
                with st.spinner("Generating comprehensive safety report..."):
                    try:
                        report_files = st.session_state.report_generator.generate_complete_report()
                        
                        if "error" in report_files:
                            st.error(report_files["error"])
                        else:
                            st.success("Safety report generated successfully!")
                            
                            # Display summary
                            summary = report_files['summary']
                            st.markdown("### 📊 Report Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Compliance Rate", f"{summary['compliance_rate']:.1f}%")
                            with col2:
                                st.metric("Total Frames", f"{summary['total_frames']:,}")
                            with col3:
                                st.metric("Violations", summary['non_compliant_frames'])
                            
                            # Download buttons
                            st.markdown("### 📥 Download Reports")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if os.path.exists(report_files['pdf_report']):
                                    with open(report_files['pdf_report'], 'rb') as file:
                                        st.download_button(
                                            label="📄 Download PDF Report",
                                            data=file,
                                            file_name=os.path.basename(report_files['pdf_report']),
                                            mime="application/pdf"
                                        )
                            
                            with col2:
                                if os.path.exists(report_files['excel_report']):
                                    with open(report_files['excel_report'], 'rb') as file:
                                        st.download_button(
                                            label="📊 Download Excel Report",
                                            data=file,
                                            file_name=os.path.basename(report_files['excel_report']),
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                    
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
        else:
            st.markdown("""
            <div class="alert-warning">
                <h3>⚠️ No Active Session</h3>
                <p>Please start a safety session to begin data collection and report generation.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

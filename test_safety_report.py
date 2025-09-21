#!/usr/bin/env python3
"""
Test script for the Safety Report Generator functionality.
This script demonstrates how to use the Safety Report Generator with sample data.
"""

import os
import sys
from datetime import datetime, timedelta
import random

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safety_report_generator import SafetyReportGenerator

def generate_sample_data(report_generator, num_frames=100):
    """
    Generate sample detection data for testing.
    
    Args:
        report_generator: SafetyReportGenerator instance
        num_frames (int): Number of sample frames to generate
    """
    print(f"Generating {num_frames} sample detection frames...")
    
    # Start a test session
    report_generator.start_session(
        session_id="test_session_001",
        zone="construction",
        model_used="test_yolov8"
    )
    
    # Generate sample data
    ppe_items = ['helmet', 'vest']
    missing_items_options = ['helmet', 'vest']
    
    for frame_num in range(num_frames):
        # Simulate realistic detection patterns
        # 70% compliance rate overall
        is_compliant = random.random() < 0.7
        
        if is_compliant:
            # Compliant case - detect all required PPE
            detected_ppe = ppe_items.copy()
            missing_items = []
        else:
            # Non-compliant case - randomly missing some PPE
            detected_ppe = []
            missing_items = random.sample(missing_items_options, random.randint(1, 2))
            
            # Sometimes detect some PPE but not all
            if random.random() < 0.3:
                detected_ppe = random.sample(ppe_items, random.randint(1, len(ppe_items)-1))
        
        # Add some confidence scores
        confidence_scores = {}
        for item in detected_ppe:
            confidence_scores[item] = round(random.uniform(0.6, 0.95), 2)
        
        # Log the detection
        report_generator.log_detection(
            frame_number=frame_num,
            is_compliant=is_compliant,
            detected_ppe=detected_ppe,
            missing_items=missing_items,
            confidence_scores=confidence_scores,
            timestamp=datetime.now() - timedelta(seconds=num_frames-frame_num),
            worker_id=f"worker_{random.randint(1, 5)}"
        )
    
    # End the session
    report_generator.end_session()
    print("Sample data generation completed!")

def test_report_generation():
    """Test the complete report generation process."""
    print("=" * 60)
    print("SAFETY REPORT GENERATOR TEST")
    print("=" * 60)
    
    # Initialize the report generator
    gemini_api_key = "AIzaSyAdMkl3-eQimVXHg2Q93vaZgBHh4pT5bFU"
    report_generator = SafetyReportGenerator(gemini_api_key)
    
    # Generate sample data
    generate_sample_data(report_generator, num_frames=50)
    
    # Get session summary
    print("\n" + "=" * 40)
    print("SESSION SUMMARY")
    print("=" * 40)
    summary = report_generator.get_session_summary()
    
    for key, value in summary.items():
        if key == 'duration':
            print(f"{key}: {value}")
        elif key in ['violation_counts', 'ppe_counts']:
            print(f"{key}:")
            for item, count in value.items():
                print(f"  - {item}: {count}")
        else:
            print(f"{key}: {value}")
    
    # Generate AI insights
    print("\n" + "=" * 40)
    print("AI-POWERED INSIGHTS")
    print("=" * 40)
    ai_insights = report_generator.generate_ai_insights(summary)
    print(ai_insights)
    
    # Generate complete report
    print("\n" + "=" * 40)
    print("GENERATING COMPLETE REPORT")
    print("=" * 40)
    
    try:
        report_files = report_generator.generate_complete_report()
        
        if "error" in report_files:
            print(f"Error generating report: {report_files['error']}")
            return False
        
        print("Report generated successfully!")
        print("\nGenerated files:")
        for file_type, file_path in report_files.items():
            if file_type not in ['summary', 'ai_insights']:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  - {file_type}: {file_path} ({file_size} bytes)")
                else:
                    print(f"  - {file_type}: {file_path} (FILE NOT FOUND)")
        
        return True
        
    except Exception as e:
        print(f"Error during report generation: {str(e)}")
        return False

def test_export_functionality():
    """Test data export functionality."""
    print("\n" + "=" * 40)
    print("TESTING EXPORT FUNCTIONALITY")
    print("=" * 40)
    
    # Create a new report generator for export testing
    gemini_api_key = "AIzaSyAdMkl3-eQimVXHg2Q93vaZgBHh4pT5bFU"
    report_generator = SafetyReportGenerator(gemini_api_key)
    
    # Generate some sample data
    generate_sample_data(report_generator, num_frames=20)
    
    # Test JSON export
    json_file = report_generator.export_detection_data('json')
    if json_file and os.path.exists(json_file):
        print(f"JSON export successful: {json_file}")
    else:
        print("JSON export failed")
    
    # Test CSV export
    csv_file = report_generator.export_detection_data('csv')
    if csv_file and os.path.exists(csv_file):
        print(f"CSV export successful: {csv_file}")
    else:
        print("CSV export failed")

def main():
    """Main test function."""
    print("Starting Safety Report Generator tests...")
    print(f"Test started at: {datetime.now()}")
    
    # Test 1: Basic report generation
    success = test_report_generation()
    
    if success:
        print("\n✅ Basic report generation test PASSED")
    else:
        print("\n❌ Basic report generation test FAILED")
    
    # Test 2: Export functionality
    test_export_functionality()
    
    print(f"\nTest completed at: {datetime.now()}")
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("Check the 'reports' directory for generated files:")
    print("  - reports/pdf/     : PDF safety reports")
    print("  - reports/excel/   : Excel reports")
    print("  - reports/charts/  : Compliance and timeline charts")
    print("  - reports/         : Raw data exports (JSON/CSV)")

if __name__ == "__main__":
    main()

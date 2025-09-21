#!/usr/bin/env python3
"""
Demo script showing how to use the Safety Report Generator.
This script demonstrates the complete workflow from session start to report generation.
"""

import os
import sys
import time
from datetime import datetime, timedelta
import random

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safety_report_generator import SafetyReportGenerator

def demo_safety_report_workflow():
    """Demonstrate the complete Safety Report Generator workflow."""
    
    print("=" * 70)
    print("SAFETY REPORT GENERATOR DEMO")
    print("=" * 70)
    print("This demo shows how to use the Safety Report Generator")
    print("to create comprehensive safety reports from PPE detection data.")
    print()
    
    # Step 1: Initialize the Safety Report Generator
    print("Step 1: Initializing Safety Report Generator...")
    gemini_api_key = "AIzaSyAdMkl3-eQimVXHg2Q93vaZgBHh4pT5bFU"
    report_generator = SafetyReportGenerator(gemini_api_key)
    print("✅ Safety Report Generator initialized successfully!")
    print()
    
    # Step 2: Start a safety session
    print("Step 2: Starting a safety monitoring session...")
    session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_generator.start_session(
        session_id=session_id,
        zone="construction",
        model_used="demo_yolov8"
    )
    print(f"✅ Safety session started: {session_id}")
    print()
    
    # Step 3: Simulate real-time detection data
    print("Step 3: Simulating real-time PPE detection...")
    print("(In a real scenario, this would be your live camera feed)")
    
    # Simulate 5 minutes of detection data (300 frames at 1 FPS)
    total_frames = 300
    ppe_items = ['helmet', 'vest']
    workers = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
    
    print(f"Processing {total_frames} frames...")
    
    for frame_num in range(total_frames):
        # Simulate realistic detection patterns
        # 80% compliance rate for this demo
        is_compliant = random.random() < 0.8
        
        if is_compliant:
            # Compliant case - detect all required PPE
            detected_ppe = ppe_items.copy()
            missing_items = []
        else:
            # Non-compliant case - randomly missing some PPE
            detected_ppe = []
            missing_items = random.sample(ppe_items, random.randint(1, 2))
            
            # Sometimes detect some PPE but not all
            if random.random() < 0.3:
                detected_ppe = random.sample(ppe_items, random.randint(1, len(ppe_items)-1))
        
        # Add confidence scores
        confidence_scores = {}
        for item in detected_ppe:
            confidence_scores[item] = round(random.uniform(0.7, 0.95), 2)
        
        # Log the detection
        report_generator.log_detection(
            frame_number=frame_num,
            is_compliant=is_compliant,
            detected_ppe=detected_ppe,
            missing_items=missing_items,
            confidence_scores=confidence_scores,
            timestamp=datetime.now() - timedelta(seconds=total_frames-frame_num),
            worker_id=random.choice(workers)
        )
        
        # Show progress every 50 frames
        if (frame_num + 1) % 50 == 0:
            print(f"  Processed {frame_num + 1}/{total_frames} frames...")
    
    print("✅ Detection simulation completed!")
    print()
    
    # Step 4: End the session
    print("Step 4: Ending the safety session...")
    report_generator.end_session()
    print("✅ Safety session ended!")
    print()
    
    # Step 5: Get session summary
    print("Step 5: Analyzing session data...")
    summary = report_generator.get_session_summary()
    
    print("📊 SESSION SUMMARY:")
    print(f"   Session ID: {summary['session_id']}")
    print(f"   Duration: {summary['duration']}")
    print(f"   Total Frames: {summary['total_frames']}")
    print(f"   Compliance Rate: {summary['compliance_rate']:.1f}%")
    print(f"   Compliant Frames: {summary['compliant_frames']}")
    print(f"   Non-Compliant Frames: {summary['non_compliant_frames']}")
    print()
    
    if summary['violation_counts']:
        print("🚨 VIOLATIONS BREAKDOWN:")
        for item, count in summary['violation_counts'].items():
            print(f"   {item.title()}: {count} violations")
        print()
    
    # Step 6: Generate AI insights
    print("Step 6: Generating AI-powered safety insights...")
    print("(This may take a few moments as we analyze the data with AI)")
    
    ai_insights = report_generator.generate_ai_insights(summary)
    print("✅ AI analysis completed!")
    print()
    
    # Step 7: Generate comprehensive report
    print("Step 7: Generating comprehensive safety report...")
    print("(Creating PDF, Excel, and chart files)")
    
    try:
        report_files = report_generator.generate_complete_report()
        
        if "error" in report_files:
            print(f"❌ Error generating report: {report_files['error']}")
            return False
        
        print("✅ Safety report generated successfully!")
        print()
        
        # Step 8: Display results
        print("Step 8: Report generation results:")
        print("=" * 50)
        
        for file_type, file_path in report_files.items():
            if file_type not in ['summary', 'ai_insights']:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"📄 {file_type.replace('_', ' ').title()}: {file_path}")
                    print(f"   Size: {file_size:,} bytes")
                else:
                    print(f"❌ {file_type.replace('_', ' ').title()}: {file_path} (NOT FOUND)")
        
        print()
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print()
        print("📁 Check the 'reports' directory for all generated files:")
        print("   - PDF Report: Professional safety report with charts")
        print("   - Excel Report: Detailed data in spreadsheet format")
        print("   - Charts: Visual compliance and timeline analysis")
        print("   - Raw Data: JSON/CSV exports for further analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during report generation: {str(e)}")
        return False

def show_ai_insights_preview():
    """Show a preview of what AI insights look like."""
    print("\n" + "=" * 50)
    print("AI INSIGHTS PREVIEW")
    print("=" * 50)
    print("The Gemini AI analyzes your safety data and provides:")
    print()
    print("🔍 KEY SAFETY INSIGHTS:")
    print("   - Identifies patterns and trends in PPE compliance")
    print("   - Highlights high-risk periods or areas")
    print("   - Compares performance against industry standards")
    print()
    print("💡 SPECIFIC RECOMMENDATIONS:")
    print("   - Actionable advice for improving safety")
    print("   - Training suggestions for specific violations")
    print("   - Process improvements for better compliance")
    print()
    print("⚠️  RISK ASSESSMENT:")
    print("   - Areas requiring immediate attention")
    print("   - Potential safety hazards")
    print("   - Compliance gaps and their impact")
    print()
    print("📈 PERFORMANCE METRICS:")
    print("   - Compliance rate analysis")
    print("   - Trend identification over time")
    print("   - Worker-specific performance insights")
    print()

def main():
    """Main demo function."""
    print("Welcome to the Safety Report Generator Demo!")
    print("This demo will show you how to create comprehensive safety reports")
    print("using AI-powered analysis of PPE detection data.")
    print()
    
    # Show AI insights preview
    show_ai_insights_preview()
    
    # Ask user if they want to continue
    response = input("Would you like to run the full demo? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print()
        success = demo_safety_report_workflow()
        
        if success:
            print("\n🎯 NEXT STEPS:")
            print("1. Open the generated PDF report to see the full analysis")
            print("2. Check the Excel file for detailed data")
            print("3. View the charts for visual insights")
            print("4. Integrate this into your existing PPE detection workflow")
            print()
            print("💡 TIP: In your actual application, the detection data")
            print("   will come from your live camera feed, not simulation!")
        else:
            print("\n❌ Demo encountered an error. Please check the error messages above.")
    else:
        print("\nDemo cancelled. Run this script again when you're ready!")
    
    print("\nThank you for trying the Safety Report Generator!")

if __name__ == "__main__":
    main()

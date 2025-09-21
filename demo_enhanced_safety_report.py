#!/usr/bin/env python3
"""
Enhanced Demo script for the improved Safety Report Generator.
This script demonstrates the enhanced PDF generation and markdown-formatted AI insights.
"""

import os
import sys
import time
from datetime import datetime, timedelta
import random

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safety_report_generator import SafetyReportGenerator

def demo_enhanced_safety_report():
    """Demonstrate the enhanced Safety Report Generator with improved PDF and AI insights."""
    
    print("=" * 80)
    print("ENHANCED SAFETY REPORT GENERATOR DEMO")
    print("=" * 80)
    print("This demo showcases the improved PDF generation and markdown-formatted AI insights.")
    print()
    
    # Step 1: Initialize the Safety Report Generator
    print("Step 1: Initializing Enhanced Safety Report Generator...")
    gemini_api_key = "AIzaSyAdMkl3-eQimVXHg2Q93vaZgBHh4pT5bFU"
    report_generator = SafetyReportGenerator(gemini_api_key)
    print("✅ Enhanced Safety Report Generator initialized successfully!")
    print()
    
    # Step 2: Start a safety session
    print("Step 2: Starting a comprehensive safety monitoring session...")
    session_id = f"enhanced_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_generator.start_session(
        session_id=session_id,
        zone="construction",
        model_used="enhanced_yolov8"
    )
    print(f"✅ Safety session started: {session_id}")
    print()
    
    # Step 3: Simulate realistic detection data with various scenarios
    print("Step 3: Simulating realistic PPE detection scenarios...")
    print("(Including various compliance patterns and violation types)")
    
    # Simulate 10 minutes of detection data (600 frames at 1 FPS)
    total_frames = 600
    ppe_items = ['helmet', 'vest']
    workers = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']
    
    print(f"Processing {total_frames} frames with realistic scenarios...")
    
    # Create realistic scenarios
    scenarios = [
        {"name": "High Compliance", "compliance_rate": 0.95, "duration": 0.2},
        {"name": "Moderate Compliance", "compliance_rate": 0.80, "duration": 0.3},
        {"name": "Low Compliance", "compliance_rate": 0.60, "duration": 0.2},
        {"name": "Variable Compliance", "compliance_rate": 0.75, "duration": 0.3}
    ]
    
    frame_count = 0
    for scenario in scenarios:
        scenario_frames = int(total_frames * scenario["duration"])
        print(f"  Simulating '{scenario['name']}' scenario ({scenario_frames} frames)...")
        
        for i in range(scenario_frames):
            # Simulate detection based on scenario
            is_compliant = random.random() < scenario["compliance_rate"]
            
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
                frame_number=frame_count,
                is_compliant=is_compliant,
                detected_ppe=detected_ppe,
                missing_items=missing_items,
                confidence_scores=confidence_scores,
                timestamp=datetime.now() - timedelta(seconds=total_frames-frame_count),
                worker_id=random.choice(workers)
            )
            
            frame_count += 1
    
    print("✅ Realistic detection simulation completed!")
    print()
    
    # Step 4: End the session
    print("Step 4: Ending the safety session...")
    report_generator.end_session()
    print("✅ Safety session ended!")
    print()
    
    # Step 5: Get session summary
    print("Step 5: Analyzing session data...")
    summary = report_generator.get_session_summary()
    
    print("📊 ENHANCED SESSION SUMMARY:")
    print(f"   Session ID: {summary['session_id']}")
    print(f"   Duration: {summary['duration']}")
    print(f"   Total Frames: {summary['total_frames']:,}")
    print(f"   Compliance Rate: {summary['compliance_rate']:.1f}%")
    print(f"   Compliant Frames: {summary['compliant_frames']:,}")
    print(f"   Non-Compliant Frames: {summary['non_compliant_frames']:,}")
    print()
    
    if summary['violation_counts']:
        print("🚨 VIOLATIONS BREAKDOWN:")
        total_violations = sum(summary['violation_counts'].values())
        for item, count in summary['violation_counts'].items():
            percentage = (count / total_violations * 100) if total_violations > 0 else 0
            print(f"   {item.title()}: {count} violations ({percentage:.1f}%)")
        print()
    
    # Step 6: Generate enhanced AI insights
    print("Step 6: Generating enhanced AI-powered safety insights...")
    print("(Using improved markdown formatting and structured analysis)")
    
    ai_insights = report_generator.generate_ai_insights(summary)
    print("✅ Enhanced AI analysis completed!")
    print()
    
    # Display AI insights preview
    print("🤖 AI INSIGHTS PREVIEW:")
    print("-" * 50)
    
    # Parse and display AI insights with proper formatting
    sections = ai_insights.split('##')
    for i, section in enumerate(sections):
        if i == 0:  # First section (before any ##)
            if section.strip():
                print(section.strip())
        else:
            # Extract section title and content
            lines = section.strip().split('\n')
            if lines:
                title = lines[0].strip()
                content = '\n'.join(lines[1:]).strip()
                
                # Display section with proper formatting
                if title:
                    print(f"\n{title}")
                    print("=" * len(title))
                if content:
                    # Convert bullet points to proper formatting
                    content_lines = content.split('\n')
                    for line in content_lines:
                        line = line.strip()
                        if line.startswith('-'):
                            print(f"  • {line[1:].strip()}")
                        elif line:
                            print(f"  {line}")
    
    print("\n" + "-" * 50)
    print()
    
    # Step 7: Generate enhanced comprehensive report
    print("Step 7: Generating enhanced comprehensive safety report...")
    print("(Creating improved PDF with better formatting, charts, and AI insights)")
    
    try:
        report_files = report_generator.generate_complete_report()
        
        if "error" in report_files:
            print(f"❌ Error generating report: {report_files['error']}")
            return False
        
        print("✅ Enhanced safety report generated successfully!")
        print()
        
        # Step 8: Display enhanced results
        print("Step 8: Enhanced report generation results:")
        print("=" * 60)
        
        for file_type, file_path in report_files.items():
            if file_type not in ['summary', 'ai_insights']:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"📄 {file_type.replace('_', ' ').title()}: {file_path}")
                    print(f"   Size: {file_size:,} bytes")
                    
                    # Show specific improvements
                    if file_type == 'pdf_report':
                        print("   ✨ Enhanced with:")
                        print("     - Professional styling and formatting")
                        print("     - Color-coded compliance status")
                        print("     - Risk level analysis")
                        print("     - Structured AI insights")
                        print("     - Executive summary")
                        print("     - Visual charts and graphs")
                else:
                    print(f"❌ {file_type.replace('_', ' ').title()}: {file_path} (NOT FOUND)")
        
        print()
        print("🎉 ENHANCED DEMO COMPLETED SUCCESSFULLY!")
        print()
        print("📁 Check the 'reports' directory for all enhanced files:")
        print("   - Enhanced PDF Report: Professional safety report with improved formatting")
        print("   - Excel Report: Detailed data in spreadsheet format")
        print("   - Enhanced Charts: Visual compliance and timeline analysis")
        print("   - Raw Data: JSON/CSV exports for further analysis")
        print()
        print("🔍 NEW FEATURES DEMONSTRATED:")
        print("   ✅ Enhanced PDF formatting with professional styling")
        print("   ✅ Color-coded compliance status indicators")
        print("   ✅ Risk level analysis for violations")
        print("   ✅ Structured AI insights with markdown formatting")
        print("   ✅ Executive summary with key metrics")
        print("   ✅ Improved visual charts and graphs")
        print("   ✅ Better data presentation and readability")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during enhanced report generation: {str(e)}")
        return False

def show_enhancement_preview():
    """Show a preview of the enhanced features."""
    print("\n" + "=" * 60)
    print("ENHANCEMENT PREVIEW")
    print("=" * 60)
    print("The enhanced Safety Report Generator now includes:")
    print()
    print("📄 ENHANCED PDF REPORTS:")
    print("   • Professional styling with custom fonts and colors")
    print("   • Executive summary with compliance status indicators")
    print("   • Color-coded risk levels (HIGH/MEDIUM/LOW)")
    print("   • Improved table formatting with better alignment")
    print("   • Structured AI insights with proper formatting")
    print("   • Visual charts and graphs integration")
    print("   • Professional footer and branding")
    print()
    print("🤖 ENHANCED AI INSIGHTS:")
    print("   • Markdown-formatted structured analysis")
    print("   • Sectioned insights (Key Insights, Risk Assessment, etc.)")
    print("   • Bullet points and emphasis formatting")
    print("   • Professional safety manager language")
    print("   • Actionable recommendations")
    print("   • Risk assessment and next steps")
    print()
    print("📊 ENHANCED DATA PRESENTATION:")
    print("   • Better compliance rate visualization")
    print("   • Risk level analysis for violations")
    print("   • Percentage breakdowns and statistics")
    print("   • Improved chart styling and readability")
    print("   • Professional color schemes")
    print()

def main():
    """Main enhanced demo function."""
    print("Welcome to the Enhanced Safety Report Generator Demo!")
    print("This demo showcases the improved PDF generation and AI insights formatting.")
    print()
    
    # Show enhancement preview
    show_enhancement_preview()
    
    # Ask user if they want to continue
    response = input("Would you like to run the enhanced demo? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print()
        success = demo_enhanced_safety_report()
        
        if success:
            print("\n🎯 NEXT STEPS:")
            print("1. Open the enhanced PDF report to see the improved formatting")
            print("2. Check the structured AI insights with markdown formatting")
            print("3. View the enhanced charts and visual analysis")
            print("4. Compare with previous reports to see improvements")
            print("5. Integrate these enhancements into your workflow")
            print()
            print("💡 TIP: The enhanced features make reports more professional")
            print("   and suitable for management presentations and safety audits!")
        else:
            print("\n❌ Enhanced demo encountered an error. Please check the error messages above.")
    else:
        print("\nEnhanced demo cancelled. Run this script again when you're ready!")
    
    print("\nThank you for trying the Enhanced Safety Report Generator!")

if __name__ == "__main__":
    main()

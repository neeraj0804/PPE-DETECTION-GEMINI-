import os
import json
import csv
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import numpy as np

class SafetyReportGenerator:
    """
    Safety Report Generator for PPE Detection System
    
    This class generates comprehensive safety reports using the Gemini API
    for analysis and insights, along with PDF and Excel export capabilities.
    """
    
    def __init__(self, gemini_api_key: str):
        """
        Initialize the Safety Report Generator.
        
        Args:
            gemini_api_key (str): Google Gemini API key for AI analysis
        """
        self.gemini_api_key = gemini_api_key
        self.configure_gemini()
        
        # Initialize data storage
        self.detection_data = []
        self.session_data = {
            'session_id': None,
            'start_time': None,
            'end_time': None,
            'total_frames': 0,
            'zone': 'construction',
            'model_used': None
        }
        
        # Create output directories
        os.makedirs('reports', exist_ok=True)
        os.makedirs('reports/pdf', exist_ok=True)
        os.makedirs('reports/excel', exist_ok=True)
        os.makedirs('reports/charts', exist_ok=True)
    
    def configure_gemini(self):
        """Configure the Gemini API."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini API configured successfully!")
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            self.model = None
    
    def start_session(self, session_id: str = None, zone: str = 'construction', model_used: str = None):
        """
        Start a new detection session.
        
        Args:
            session_id (str): Unique session identifier
            zone (str): Work zone type
            model_used (str): Model being used for detection
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_data.update({
            'session_id': session_id,
            'start_time': datetime.now(),
            'zone': zone,
            'model_used': model_used
        })
        
        print(f"Started new session: {session_id}")
    
    def end_session(self):
        """End the current detection session."""
        self.session_data['end_time'] = datetime.now()
        self.session_data['total_frames'] = len(self.detection_data)
        print(f"Ended session: {self.session_data['session_id']}")
    
    def log_detection(self, frame_number: int, is_compliant: bool, detected_ppe: List[str], 
                     missing_items: List[str], confidence_scores: Dict[str, float] = None,
                     timestamp: datetime = None, worker_id: str = None):
        """
        Log a detection result.
        
        Args:
            frame_number (int): Frame number in the sequence
            is_compliant (bool): Whether the detection was compliant
            detected_ppe (List[str]): List of detected PPE items
            missing_items (List[str]): List of missing PPE items
            confidence_scores (Dict[str, float]): Confidence scores for each detection
            timestamp (datetime): Timestamp of detection
            worker_id (str): Optional worker identifier
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        detection_record = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'is_compliant': is_compliant,
            'detected_ppe': detected_ppe.copy(),
            'missing_items': missing_items.copy(),
            'confidence_scores': confidence_scores or {},
            'worker_id': worker_id,
            'session_id': self.session_data['session_id']
        }
        
        self.detection_data.append(detection_record)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        if not self.detection_data:
            return {}
        
        total_frames = len(self.detection_data)
        compliant_frames = sum(1 for d in self.detection_data if d['is_compliant'])
        non_compliant_frames = total_frames - compliant_frames
        
        # Calculate compliance rate
        compliance_rate = (compliant_frames / total_frames * 100) if total_frames > 0 else 0
        
        # Count violations by type
        violation_counts = defaultdict(int)
        for detection in self.detection_data:
            for missing_item in detection['missing_items']:
                violation_counts[missing_item] += 1
        
        # Count detected PPE
        ppe_counts = defaultdict(int)
        for detection in self.detection_data:
            for ppe_item in detection['detected_ppe']:
                ppe_counts[ppe_item] += 1
        
        # Calculate session duration
        if self.session_data['start_time'] and self.session_data['end_time']:
            duration = self.session_data['end_time'] - self.session_data['start_time']
        else:
            duration = timedelta(0)
        
        return {
            'session_id': self.session_data['session_id'],
            'start_time': self.session_data['start_time'],
            'end_time': self.session_data['end_time'],
            'duration': duration,
            'total_frames': total_frames,
            'compliant_frames': compliant_frames,
            'non_compliant_frames': non_compliant_frames,
            'compliance_rate': compliance_rate,
            'violation_counts': dict(violation_counts),
            'ppe_counts': dict(ppe_counts),
            'zone': self.session_data['zone'],
            'model_used': self.session_data['model_used']
        }
    
    def generate_ai_insights(self, summary: Dict[str, Any]) -> str:
        """
        Generate AI-powered insights using Gemini API with markdown formatting.
        
        Args:
            summary (Dict[str, Any]): Session summary data
            
        Returns:
            str: AI-generated insights and recommendations in markdown format
        """
        if self.model is None:
            return "AI insights unavailable - Gemini API not configured properly."
        
        try:
            prompt = f"""
            As a safety expert, analyze the following PPE detection data and provide insights and recommendations in markdown format:
            
            Session Summary:
            - Session ID: {summary.get('session_id', 'N/A')}
            - Duration: {summary.get('duration', 'N/A')}
            - Total Frames Analyzed: {summary.get('total_frames', 0)}
            - Compliance Rate: {summary.get('compliance_rate', 0):.1f}%
            - Work Zone: {summary.get('zone', 'N/A')}
            
            Violation Statistics:
            {json.dumps(summary.get('violation_counts', {}), indent=2)}
            
            PPE Detection Statistics:
            {json.dumps(summary.get('ppe_counts', {}), indent=2)}
            
            Please provide your analysis in the following markdown format:
            
            ## 🔍 Key Safety Insights
            - [List 3-5 key insights about the safety data]
            
            ## ⚠️ Risk Assessment
            - [Identify high-risk areas and concerns]
            
            ## 💡 Specific Recommendations
            - [List actionable recommendations for improvement]
            
            ## 📈 Performance Analysis
            - [Analyze trends and patterns in the data]
            
            ## 🎯 Next Steps
            - [Outline immediate and long-term actions needed]
            
            Use bullet points, bold text for emphasis, and clear section headers. Make it professional and actionable for safety managers.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating AI insights: {str(e)}"
    
    def _format_ai_insights_for_pdf(self, ai_insights: str) -> List[Dict[str, str]]:
        """
        Format AI insights for PDF display with proper styling.
        
        Args:
            ai_insights (str): Raw AI insights text
            
        Returns:
            List[Dict[str, str]]: Formatted sections for PDF
        """
        formatted_sections = []
        lines = ai_insights.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Headers (## or ###)
            if line.startswith('##'):
                content = line.replace('#', '').strip()
                # Remove emoji and clean up
                content = content.split(' ', 1)[-1] if ' ' in content else content
                formatted_sections.append({
                    'type': 'header',
                    'content': content
                })
            # Bullet points
            elif line.startswith('-') or line.startswith('•'):
                content = line[1:].strip()
                formatted_sections.append({
                    'type': 'bullet',
                    'content': content
                })
            # Highlighted text (bold or important)
            elif line.startswith('**') and line.endswith('**'):
                content = line.replace('**', '').strip()
                formatted_sections.append({
                    'type': 'highlight',
                    'content': content
                })
            # Regular paragraphs
            else:
                # Clean up markdown formatting
                content = line.replace('**', '').replace('*', '').strip()
                if content:
                    formatted_sections.append({
                        'type': 'paragraph',
                        'content': content
                    })
        
        return formatted_sections
    
    def create_compliance_chart(self, summary: Dict[str, Any], output_path: str = None):
        """
        Create a compliance rate chart.
        
        Args:
            summary (Dict[str, Any]): Session summary data
            output_path (str): Path to save the chart
        """
        if output_path is None:
            output_path = f"reports/charts/compliance_chart_{self.session_data['session_id']}.png"
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Compliance pie chart
        compliant = summary.get('compliant_frames', 0)
        non_compliant = summary.get('non_compliant_frames', 0)
        
        if compliant + non_compliant > 0:
            ax1.pie([compliant, non_compliant], 
                   labels=['Compliant', 'Non-Compliant'],
                   colors=['#2ecc71', '#e74c3c'],
                   autopct='%1.1f%%',
                   startangle=90)
            ax1.set_title('Overall Compliance Rate')
        
        # Violations bar chart
        violations = summary.get('violation_counts', {})
        if violations:
            items = list(violations.keys())
            counts = list(violations.values())
            
            bars = ax2.bar(items, counts, color='#e74c3c')
            ax2.set_title('Violations by PPE Type')
            ax2.set_xlabel('Missing PPE Items')
            ax2.set_ylabel('Number of Violations')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_timeline_chart(self, output_path: str = None):
        """
        Create a timeline chart showing compliance over time.
        
        Args:
            output_path (str): Path to save the chart
        """
        if not self.detection_data:
            return None
        
        if output_path is None:
            output_path = f"reports/charts/timeline_chart_{self.session_data['session_id']}.png"
        
        # Group data by time windows (every 10 frames)
        window_size = 10
        windows = []
        compliance_rates = []
        
        for i in range(0, len(self.detection_data), window_size):
            window_data = self.detection_data[i:i+window_size]
            if window_data:
                compliant_count = sum(1 for d in window_data if d['is_compliant'])
                compliance_rate = (compliant_count / len(window_data)) * 100
                windows.append(i // window_size)
                compliance_rates.append(compliance_rate)
        
        # Create timeline chart
        plt.figure(figsize=(12, 6))
        plt.plot(windows, compliance_rates, marker='o', linewidth=2, markersize=4)
        plt.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Target (80%)')
        plt.xlabel('Time Window (every 10 frames)')
        plt.ylabel('Compliance Rate (%)')
        plt.title('PPE Compliance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_pdf_report(self, summary: Dict[str, Any], ai_insights: str, 
                           compliance_chart_path: str = None, timeline_chart_path: str = None) -> str:
        """
        Generate a comprehensive PDF report.
        
        Args:
            summary (Dict[str, Any]): Session summary data
            ai_insights (str): AI-generated insights
            compliance_chart_path (str): Path to compliance chart
            timeline_chart_path (str): Path to timeline chart
            
        Returns:
            str: Path to generated PDF report
        """
        # Create PDF file
        pdf_path = f"reports/pdf/safety_report_{self.session_data['session_id']}.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                               rightMargin=72, leftMargin=72, 
                               topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=8,
            backColor=colors.lightgrey
        )
        
        subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            fontName='Helvetica'
        )
        
        bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            leftIndent=20,
            fontName='Helvetica'
        )
        
        # Title and Header
        story.append(Paragraph("PPE SAFETY COMPLIANCE REPORT", title_style))
        story.append(Spacer(1, 10))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", body_style))
        story.append(Paragraph(f"<b>Report ID:</b> {summary.get('session_id', 'N/A')}", body_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", header_style))
        
        compliance_rate = summary.get('compliance_rate', 0)
        total_frames = summary.get('total_frames', 0)
        non_compliant = summary.get('non_compliant_frames', 0)
        
        # Compliance status with color coding
        if compliance_rate >= 90:
            status_color = colors.green
            status_text = "EXCELLENT"
        elif compliance_rate >= 80:
            status_color = colors.orange
            status_text = "GOOD"
        elif compliance_rate >= 70:
            status_color = colors.red
            status_text = "NEEDS IMPROVEMENT"
        else:
            status_color = colors.darkred
            status_text = "CRITICAL"
        
        story.append(Paragraph(f"<b>Overall Compliance Rate:</b> {compliance_rate:.1f}%", body_style))
        story.append(Paragraph(f"<b>Safety Status:</b> <font color='{status_color.hexval()}'>{status_text}</font>", body_style))
        story.append(Paragraph(f"<b>Total Frames Analyzed:</b> {total_frames:,}", body_style))
        story.append(Paragraph(f"<b>Violations Detected:</b> {non_compliant:,}", body_style))
        story.append(Spacer(1, 20))
        
        # Session Information
        story.append(Paragraph("SESSION DETAILS", header_style))
        session_info = [
            ['Session ID:', summary.get('session_id', 'N/A')],
            ['Start Time:', summary.get('start_time', 'N/A').strftime('%Y-%m-%d %H:%M:%S') if summary.get('start_time') else 'N/A'],
            ['End Time:', summary.get('end_time', 'N/A').strftime('%Y-%m-%d %H:%M:%S') if summary.get('end_time') else 'N/A'],
            ['Duration:', str(summary.get('duration', 'N/A'))],
            ['Work Zone:', summary.get('zone', 'N/A').title()],
            ['Model Used:', summary.get('model_used', 'N/A')]
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 4*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(session_table)
        story.append(Spacer(1, 20))
        
        # Compliance Summary
        story.append(Paragraph("COMPLIANCE ANALYSIS", header_style))
        compliance_data = [
            ['Metric', 'Count', 'Percentage'],
            ['Total Frames Analyzed', f"{summary.get('total_frames', 0):,}", '100.0%'],
            ['Compliant Frames', f"{summary.get('compliant_frames', 0):,}", f"{compliance_rate:.1f}%"],
            ['Non-Compliant Frames', f"{summary.get('non_compliant_frames', 0):,}", f"{100-compliance_rate:.1f}%"]
        ]
        
        compliance_table = Table(compliance_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        compliance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(compliance_table)
        story.append(Spacer(1, 20))
        
        # Violations Breakdown
        violations = summary.get('violation_counts', {})
        if violations:
            story.append(Paragraph("VIOLATIONS BREAKDOWN", header_style))
            
            # Calculate violation percentages
            total_violations = sum(violations.values())
            violation_data = [['PPE Item', 'Violations', 'Percentage', 'Risk Level']]
            
            for item, count in violations.items():
                percentage = (count / total_violations * 100) if total_violations > 0 else 0
                if percentage >= 50:
                    risk_level = "HIGH"
                elif percentage >= 25:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                
                violation_data.append([
                    item.title().replace('_', ' '),
                    str(count),
                    f"{percentage:.1f}%",
                    risk_level
                ])
            
            violation_table = Table(violation_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch])
            violation_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (3, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Color code risk levels
                ('TEXTCOLOR', (3, 1), (3, -1), colors.red if any('HIGH' in row[3] for row in violation_data[1:]) else colors.black),
            ]))
            story.append(violation_table)
            story.append(Spacer(1, 20))
        
        # Add charts if available
        if compliance_chart_path and os.path.exists(compliance_chart_path):
            story.append(Paragraph("VISUAL ANALYSIS", header_style))
            story.append(Image(compliance_chart_path, width=6*inch, height=3*inch))
            story.append(Spacer(1, 10))
        
        if timeline_chart_path and os.path.exists(timeline_chart_path):
            story.append(Image(timeline_chart_path, width=6*inch, height=3*inch))
            story.append(Spacer(1, 20))
        
        # AI Insights with better formatting
        story.append(Paragraph("AI-POWERED SAFETY ANALYSIS", header_style))
        
        # Parse and format AI insights with markdown-like structure
        formatted_insights = self._format_ai_insights_for_pdf(ai_insights)
        
        for section in formatted_insights:
            if section['type'] == 'header':
                story.append(Paragraph(section['content'], subheader_style))
            elif section['type'] == 'bullet':
                story.append(Paragraph(f"• {section['content']}", bullet_style))
            elif section['type'] == 'paragraph':
                story.append(Paragraph(section['content'], body_style))
            elif section['type'] == 'highlight':
                highlight_style = ParagraphStyle(
                    'HighlightStyle',
                    parent=body_style,
                    backColor=colors.yellow,
                    borderWidth=1,
                    borderColor=colors.orange,
                    borderPadding=8
                )
                story.append(Paragraph(section['content'], highlight_style))
        
        story.append(Spacer(1, 20))
        
        # Recommendations section
        story.append(Paragraph("RECOMMENDATIONS", header_style))
        story.append(Paragraph("Based on the AI analysis above, the following actions are recommended:", body_style))
        story.append(Spacer(1, 10))
        
        # Add page break before footer
        story.append(Spacer(1, 20))
        
        # Footer
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
        story.append(Paragraph("Generated by PPE Safety Report Generator | AI-Powered Safety Analysis", footer_style))
        
        # Build PDF
        doc.build(story)
        return pdf_path
    
    def generate_excel_report(self, summary: Dict[str, Any]) -> str:
        """
        Generate an Excel report with detailed data.
        
        Args:
            summary (Dict[str, Any]): Session summary data
            
        Returns:
            str: Path to generated Excel report
        """
        excel_path = f"reports/excel/safety_report_{self.session_data['session_id']}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([
                ['Session ID', summary.get('session_id', 'N/A')],
                ['Start Time', summary.get('start_time', 'N/A')],
                ['End Time', summary.get('end_time', 'N/A')],
                ['Duration', str(summary.get('duration', 'N/A'))],
                ['Total Frames', summary.get('total_frames', 0)],
                ['Compliant Frames', summary.get('compliant_frames', 0)],
                ['Non-Compliant Frames', summary.get('non_compliant_frames', 0)],
                ['Compliance Rate', f"{summary.get('compliance_rate', 0):.1f}%"],
                ['Work Zone', summary.get('zone', 'N/A')],
                ['Model Used', summary.get('model_used', 'N/A')]
            ], columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Violations sheet
            violations = summary.get('violation_counts', {})
            if violations:
                violations_df = pd.DataFrame([
                    {'PPE Item': item, 'Violations': count}
                    for item, count in violations.items()
                ])
                violations_df.to_excel(writer, sheet_name='Violations', index=False)
            
            # Detailed detections sheet
            if self.detection_data:
                detections_df = pd.DataFrame(self.detection_data)
                detections_df['timestamp'] = detections_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                detections_df.to_excel(writer, sheet_name='Detailed Detections', index=False)
        
        return excel_path
    
    def generate_complete_report(self) -> Dict[str, str]:
        """
        Generate a complete safety report with all components.
        
        Returns:
            Dict[str, str]: Dictionary containing paths to generated files
        """
        if not self.detection_data:
            return {"error": "No detection data available for report generation"}
        
        # Get session summary
        summary = self.get_session_summary()
        
        # Generate AI insights
        ai_insights = self.generate_ai_insights(summary)
        
        # Create charts
        compliance_chart_path = self.create_compliance_chart(summary)
        timeline_chart_path = self.create_timeline_chart()
        
        # Generate reports
        pdf_path = self.generate_pdf_report(summary, ai_insights, compliance_chart_path, timeline_chart_path)
        excel_path = self.generate_excel_report(summary)
        
        return {
            "pdf_report": pdf_path,
            "excel_report": excel_path,
            "compliance_chart": compliance_chart_path,
            "timeline_chart": timeline_chart_path,
            "summary": summary,
            "ai_insights": ai_insights
        }
    
    def clear_session_data(self):
        """Clear all session data."""
        self.detection_data.clear()
        self.session_data = {
            'session_id': None,
            'start_time': None,
            'end_time': None,
            'total_frames': 0,
            'zone': 'construction',
            'model_used': None
        }
        print("Session data cleared.")
    
    def export_detection_data(self, format: str = 'json') -> str:
        """
        Export detection data in specified format.
        
        Args:
            format (str): Export format ('json', 'csv')
            
        Returns:
            str: Path to exported file
        """
        if not self.detection_data:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'json':
            file_path = f"reports/detection_data_{timestamp}.json"
            with open(file_path, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                export_data = []
                for record in self.detection_data:
                    export_record = record.copy()
                    export_record['timestamp'] = record['timestamp'].isoformat()
                    export_data.append(export_record)
                json.dump(export_data, f, indent=2)
        
        elif format.lower() == 'csv':
            file_path = f"reports/detection_data_{timestamp}.csv"
            df = pd.DataFrame(self.detection_data)
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df.to_csv(file_path, index=False)
        
        return file_path

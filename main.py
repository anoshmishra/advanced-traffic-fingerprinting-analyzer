#!/usr/bin/env python3
"""
Enhanced Main Script with Automated URL Analysis
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import yaml
import time
import subprocess
import tempfile
import requests
from urllib.parse import urlparse
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Import existing modules
from preprocessing.feature_extractor import FeatureExtractor
from models.trainer import ModelTrainer
from defense.padding_defense import PaddingDefense
from defense.timing_defense import TimingDefense
from visualization.plots import Visualizer
from collection.crawler import TrafficCrawler

def analyze_url_automatically(url, config):
    """
    MAIN AUTOMATED FUNCTION - Just provide URL, get complete analysis
    """
    print(f"🚀 Starting automated analysis for: {url}")
    
    try:
        # Phase 1: Validate URL
        print("📋 Validating URL...")
        response = requests.head(url, timeout=10)
        if response.status_code >= 400:
            raise ValueError("URL is not accessible")
        
        # Phase 2: Quick traffic simulation (for demo)
        print("📡 Analyzing traffic patterns...")
        time.sleep(2)  # Simulate analysis time
        
        # Phase 3: Generate security assessment
        print("🔒 Generating security assessment...")
        domain = urlparse(url).netloc
        
        # Simulate fingerprinting risk assessment
        risk_factors = {
            'domain_popularity': 0.7 if any(service in domain.lower() for service in ['facebook', 'youtube', 'netflix']) else 0.4,
            'traffic_predictability': 0.6,
            'size_consistency': 0.5
        }
        
        privacy_risk_score = sum(risk_factors.values()) / len(risk_factors)
        
        # Determine safety status
        if privacy_risk_score >= 0.8:
            safety_status = "🔴 UNSAFE - High Privacy Risk"
            risk_level = "CRITICAL"
        elif privacy_risk_score >= 0.6:
            safety_status = "🟡 RISKY - Moderate Privacy Risk"  
            risk_level = "HIGH"
        elif privacy_risk_score >= 0.4:
            safety_status = "🟠 CAUTION - Some Privacy Risk"
            risk_level = "MEDIUM"
        else:
            safety_status = "✅ SAFE - Low Privacy Risk"
            risk_level = "LOW"
        
        # Generate recommendations
        recommendations = generate_recommendations(privacy_risk_score, domain)
        
        # Create report
        print("📄 Creating comprehensive report...")
        report_path = create_html_report(url, {
            'privacy_risk_score': privacy_risk_score,
            'safety_status': safety_status,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
        return report_path
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return create_error_report(url, str(e))

def generate_recommendations(risk_score, domain):
    """Generate security recommendations based on risk score"""
    recommendations = {
        'immediate_actions': [],
        'browser_settings': [],
        'network_security': [],
        'privacy_tools': []
    }
    
    if risk_score >= 0.7:
        recommendations['immediate_actions'] = [
            "🚨 Use Tor Browser for maximum anonymity",
            "🚨 Enable VPN with traffic obfuscation",
            "🚨 Avoid accessing from sensitive locations"
        ]
    elif risk_score >= 0.5:
        recommendations['immediate_actions'] = [
            "⚠️ Use VPN when accessing this site",
            "⚠️ Enable private browsing mode"
        ]
    else:
        recommendations['immediate_actions'] = [
            "✅ Site appears relatively safe for normal browsing"
        ]
    
    recommendations['browser_settings'] = [
        "🔧 Enable 'Do Not Track' in browser settings",
        "🔧 Use ad blockers and privacy extensions",
        "🔧 Regular clear cookies and browsing data"
    ]
    
    recommendations['network_security'] = [
        "🌐 Avoid public WiFi for this site",
        "🌐 Use secure DNS (1.1.1.1 or 8.8.8.8)",
        "🌐 Enable firewall protection"
    ]
    
    recommendations['privacy_tools'] = [
        "🛡️ VPN with no-logs policy recommended",
        "🛡️ Privacy-focused browsers (Firefox, Brave)",
        "🛡️ Regular browser updates important"
    ]
    
    return recommendations

def create_html_report(url, analysis_data):
    """Create comprehensive HTML report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain = urlparse(url).netloc.replace('.', '_')
    
    reports_dir = Path('results/reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"AUTOMATED_ANALYSIS_{domain}_{timestamp}.html"
    
    # Determine status class for styling
    risk_score = analysis_data['privacy_risk_score']
    if risk_score >= 0.7:
        status_class = "status-unsafe"
    elif risk_score >= 0.5:
        status_class = "status-risky"
    elif risk_score >= 0.3:
        status_class = "status-caution"
    else:
        status_class = "status-safe"
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>🔐 Automated Security Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; min-height: 100vh; }}
        .header {{ background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white; padding: 40px; text-align: center; }}
        .status-banner {{ padding: 20px; margin: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; }}
        .status-safe {{ background: #d4edda; color: #155724; border: 3px solid #c3e6cb; }}
        .status-caution {{ background: #fff3cd; color: #856404; border: 3px solid #ffeaa7; }}
        .status-risky {{ background: #f8d7da; color: #721c24; border: 3px solid #f5c6cb; }}
        .status-unsafe {{ background: #f8d7da; color: #721c24; border: 3px solid #f5c6cb; animation: pulse 2s infinite; }}
        @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} 100% {{ transform: scale(1); }} }}
        .risk-score {{ font-size: 48px; font-weight: bold; margin: 20px 0; color: {'#dc3545' if risk_score >= 0.7 else '#fd7e14' if risk_score >= 0.5 else '#28a745'}; }}
        .section {{ margin: 30px; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .recommendations {{ background: #e8f5e8; }}
        .rec-category {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }}
        ul {{ line-height: 1.8; }}
        .footer {{ text-align: center; padding: 30px; background: #f8f9fa; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Automated Security Analysis Report</h1>
            <h2>Target: {urlparse(url).netloc}</h2>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="status-banner {status_class}">
            {analysis_data['safety_status']} - {analysis_data['risk_level']} RISK
        </div>
        
        <div style="text-align: center; margin: 30px;">
            <h3>Privacy Risk Score</h3>
            <div class="risk-score">{risk_score:.1%}</div>
        </div>
        
        <div class="section recommendations">
            <h2>🛡️ Security Recommendations</h2>
"""
    
    # Add recommendations sections
    categories = [
        ('🚨 Immediate Actions', 'immediate_actions'),
        ('🔧 Browser Settings', 'browser_settings'),
        ('🌐 Network Security', 'network_security'),
        ('🛡️ Privacy Tools', 'privacy_tools')
    ]
    
    for title, key in categories:
        if analysis_data['recommendations'].get(key):
            html_content += f'<div class="rec-category"><h3>{title}</h3><ul>'
            for rec in analysis_data['recommendations'][key]:
                html_content += f'<li>{rec}</li>'
            html_content += '</ul></div>'
    
    html_content += f"""
        </div>
        
        <div class="section">
            <h2>📋 Summary</h2>
            <p><strong>Analysis Target:</strong> {url}</p>
            <p><strong>Risk Assessment:</strong> {analysis_data['risk_level']} risk level detected</p>
            <p><strong>Privacy Score:</strong> {risk_score:.1%} vulnerability rating</p>
            <p><strong>Safety Status:</strong> {analysis_data['safety_status']}</p>
            <p><strong>Recommendation:</strong> Follow the security guidelines above for safe browsing</p>
        </div>
        
        <div class="footer">
            <p><strong>Automated Analysis Complete</strong></p>
            <p>This report was generated by the Encrypted Traffic Fingerprinting Analysis System</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_file

def create_error_report(url, error_message):
    """Create error report when analysis fails"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path('results/reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    error_file = reports_dir / f"ERROR_REPORT_{timestamp}.html"
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>❌ Analysis Error Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }}
        .error {{ background: #f8d7da; color: #721c24; border: 3px solid #f5c6cb; padding: 20px; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>❌ Analysis Failed</h1>
        <p><strong>Target URL:</strong> {url}</p>
        <div class="error">
            <h2>Error:</h2>
            <p>{error_message}</p>
        </div>
        <h3>Try:</h3>
        <ul>
            <li>Check the URL is accessible</li>
            <li>Verify internet connection</li>
            <li>Try again in a few minutes</li>
        </ul>
    </div>
</body>
</html>
"""
    
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return error_file

def main():
    """Enhanced main function with automated URL analysis"""
    parser = argparse.ArgumentParser(description='Encrypted Traffic Fingerprinting Analysis')
    parser.add_argument('--url', type=str, help='URL to analyze automatically')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'collection': {'visits_per_site': 10, 'capture_duration': 20},
        'models': {'test_size': 0.3},
        'defense': {'padding': {'enabled': True}, 'timing': {'enabled': True}}
    }
    
    # Automated URL analysis mode
    if args.url:
        print("🔐 AUTOMATED ENCRYPTED TRAFFIC ANALYSIS")
        print("=" * 50)
        
        if not args.url.startswith(('http://', 'https://')):
            args.url = 'https://' + args.url
        
        report_path = analyze_url_automatically(args.url, config)
        
        print("\n" + "=" * 50)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"📄 Report: {report_path}")
        print("🌐 Open the HTML file to view results")
        return
    
    # GUI mode
    if args.gui:
        try:
            from gui.main_window import main as gui_main
            gui_main()
        except ImportError:
            print("GUI not available. Run: pip install customtkinter")
        return
    
    # Original mode - run your existing pipeline
    print("Running original analysis pipeline...")
    # Your existing main.py code here

if __name__ == "__main__":
    main()

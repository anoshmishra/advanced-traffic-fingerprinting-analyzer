#!/usr/bin/env python3
"""
Launch script for the GUI application
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Import and run GUI
from gui.main_window import main

if __name__ == "__main__":
    print("🔐 Launching Encrypted Traffic Fingerprinting GUI...")
    print("📊 Advanced Analysis Suite")
    print("=" * 50)
    
    # Check if results exist
    results_dir = current_dir / 'results' / 'reports'
    if not results_dir.exists() or not list(results_dir.glob('*_training_results.joblib')):
        print("⚠️  No existing results found.")
        print("💡 Run 'python3 main.py' first to generate baseline results.")
        print("   Or the GUI will use sample data for demonstration.")
        print()
    
    main()

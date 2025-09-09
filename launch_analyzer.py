#!/usr/bin/env python3

import os
import sys
from pathlib import Path

print("🔐 Launching Advanced Traffic Analyzer...")
print("📊 Professional Analysis Suite")
print("=" * 50)

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.gui.advanced_traffic_analyzer import main
    main()
except Exception as e:
    print(f"Error starting application: {e}")
    import traceback
    traceback.print_exc()

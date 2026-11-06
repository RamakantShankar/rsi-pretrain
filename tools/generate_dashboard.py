#!/usr/bin/env python3
"""
Mirendil Laboratory - Executive Experiment Dashboard Generator
Shortcut that invokes tools/generate_interactive_dashboard.py.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.generate_interactive_dashboard import main

if __name__ == "__main__":
    main()

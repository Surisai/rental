#!/usr/bin/env python3
"""
Entry point for running the province analysis.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from province_analysis.main import main

if __name__ == "__main__":
    main()
"""
Configuration module for province analysis.
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Input path
INPUT_PATH = BASE_DIR / "data" / "processed" / "rentals_clean.csv"

# Main output directory
MAIN_OUTPUT_DIR = BASE_DIR / "outputs" / "province_analysis"

# Structured subdirectories
OUTPUT_DIRS = {
    'main': MAIN_OUTPUT_DIR,
    'reports': MAIN_OUTPUT_DIR / 'reports',
    'data': MAIN_OUTPUT_DIR / 'data',
    'visualizations': MAIN_OUTPUT_DIR / 'visualizations',
    'provinces': MAIN_OUTPUT_DIR / 'data' / 'provinces',
    'province_charts': MAIN_OUTPUT_DIR / 'visualizations' / 'province_distributions',
    'summary_charts': MAIN_OUTPUT_DIR / 'visualizations' / 'summary_charts'
}

# Create all directories
for dir_path in OUTPUT_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)
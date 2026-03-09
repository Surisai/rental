"""
Configuration module for the rental data cleaning pipeline.
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parents[2]

# File paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "rentals.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "rentals_clean.csv"
LOG_PATH = BASE_DIR / "logs" / "data_cleaning_log.txt"

# Create directories if they don't exist
CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
"""
Data loading and saving utilities.
"""

import pandas as pd
from .config import RAW_DATA_PATH, CLEAN_DATA_PATH


def load_raw_data(filepath):
    """Load and inspect raw data"""
    print("="*60)
    print("LOADING RAW DATA")
    print("="*60)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Log initial state
        log_details = [
            f"Initial rows: {df.shape[0]:,}",
            f"Initial columns: {df.shape[1]}",
            f"Columns: {', '.join(df.columns.tolist())}"
        ]
        
        return df, log_details
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


def save_clean_data(df, filepath):
    """Save cleaned data to CSV"""
    print("\n" + "="*60)
    print("SAVING CLEANED DATA")
    print("="*60)
    
    try:
        df.to_csv(filepath, index=False)
        print(f"✓ Cleaned data saved to: {filepath}")
        print(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        return [f"Data saved to: {filepath}", f"File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB"]
    except Exception as e:
        print(f"✗ Error saving data: {e}")
        raise
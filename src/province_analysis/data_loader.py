"""
Data loading utilities for province analysis.
"""

import pandas as pd
from .config import INPUT_PATH


def load_data(filepath):
    """Load CSV with error handling"""
    print("="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise
"""
Data cleaning utilities for province analysis.
"""

import pandas as pd


def clean_data(df):
    """Clean data - SIMPLIFIED for pre-cleaned data"""
    print("\n" + "="*60)
    print("STEP 2: DATA PREPARATION (PRE-CLEANED DATA)")
    print("="*60)
    
    print("Data is already clean, performing quick checks...")
    
    df_clean = df.copy()
    
    # --- Quick validation ---
    print("\n1. VALIDATING DATA...")
    original_rows = len(df_clean)
    
    # Ensure price is valid
    df_clean = df_clean[(df_clean['price'] > 0) & (df_clean['price'].notna())]
    
    # Ensure beds is numeric if it exists
    if 'beds' in df_clean.columns and df_clean['beds'].dtype == 'object':
        df_clean['beds'] = pd.to_numeric(df_clean['beds'], errors='coerce')
    
    final_rows = len(df_clean)
    
    if original_rows != final_rows:
        print(f"  ⚠ Removed {original_rows - final_rows} invalid rows")
    else:
        print(f"  ✓ All {final_rows:,} rows valid")
    
    return df_clean
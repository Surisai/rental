"""
Text cleaning utilities.
"""

import pandas as pd
import numpy as np


def normalize_text_columns(df):
    """Standardize text columns (lowercase, strip whitespace)"""
    print("\n" + "="*60)
    print("NORMALIZING TEXT COLUMNS")
    print("="*60)
    
    text_columns = ['city', 'province', 'type', 'furnishing', 'smoking', 'lease_term']
    
    changes_log = []
    for col in text_columns:
        if col in df.columns:
            before_sample = df[col].head(3).tolist() if len(df) > 0 else []
            df[col] = df[col].astype(str).str.strip().str.lower()
            after_sample = df[col].head(3).tolist() if len(df) > 0 else []
            
            if before_sample != after_sample:
                changes_log.append(f"  {col}: Normalized")
            print(f"✓ Normalized: {col}")
    
    log_details = ["Text columns normalized to lowercase and stripped whitespace:"]
    log_details.extend(changes_log if changes_log else ["  No changes needed"])
    
    # Special handling for province names
    if 'province' in df.columns:
        province_mapping = {
            'bc': 'british columbia',
            'b.c.': 'british columbia',
            'britishcolumbia': 'british columbia',
            'on': 'ontario',
            'ont.': 'ontario',
            'qc': 'quebec',
            'queb.': 'quebec',
            'ab': 'alberta',
            'alb.': 'alberta',
            'mb': 'manitoba',
            'man.': 'manitoba',
            'sk': 'saskatchewan',
            'sask.': 'saskatchewan',
            'ns': 'nova scotia',
            'nb': 'new brunswick',
            'nl': 'newfoundland and labrador',
            'nfld': 'newfoundland and labrador',
            'pe': 'prince edward island',
            'pei': 'prince edward island',
            'yt': 'yukon',
            'nt': 'northwest territories',
            'nwt': 'northwest territories',
            'nu': 'nunavut'
        }
        
        original_provinces = df['province'].unique()[:5]
        df['province'] = df['province'].replace(province_mapping)
        cleaned_provinces = df['province'].unique()[:5]
        
        log_details.append("\nProvince name standardization:")
        log_details.append(f"  Original samples: {original_provinces}")
        log_details.append(f"  Cleaned samples: {cleaned_provinces}")
        print(f"✓ Standardized province names")
    
    return df, log_details


def clean_pet_columns(df):
    """Standardize pet columns (cats, dogs)"""
    print("\n" + "="*60)
    print("CLEANING PET COLUMNS")
    print("="*60)
    
    pet_columns = ['cats', 'dogs']
    log_details = []
    
    for col in pet_columns:
        if col in df.columns:
            original_dtype = str(df[col].dtype)
            original_sample = df[col].head(5).tolist()
            
            # Convert to string and standardize
            df[col] = df[col].astype(str).str.lower().str.strip()
            
            # Map various representations to yes/no
            yes_patterns = ['true', 't', 'yes', 'y', '1', 'allowed', 'permitted']
            no_patterns = ['false', 'f', 'no', 'n', '0', 'not allowed', 'not permitted']
            
            def standardize_pet(value):
                value_str = str(value).lower().strip()
                if any(pattern in value_str for pattern in yes_patterns):
                    return 'yes'
                elif any(pattern in value_str for pattern in no_patterns):
                    return 'no'
                elif value_str in ['nan', 'none', '']:
                    return 'unknown'
                else:
                    return value_str
            
            df[col] = df[col].apply(standardize_pet)
            
            # Fill NaN with 'unknown'
            df[col] = df[col].fillna('unknown')
            
            distribution = df[col].value_counts()
            
            print(f"✓ Cleaned: {col}")
            print(f"  Distribution: {dict(distribution)}")
            
            log_details.append(f"\n{col.upper()}:")
            log_details.append(f"  Original dtype: {original_dtype}")
            log_details.append(f"  Original samples: {original_sample}")
            log_details.append(f"  Distribution: {dict(distribution)}")
    
    return df, log_details
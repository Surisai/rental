"""
Numeric cleaning utilities.
"""

import pandas as pd
import numpy as np
import re


def clean_price_column(df):
    """Clean and convert price column to numeric"""
    print("\n" + "="*60)
    print("CLEANING PRICE COLUMN")
    print("="*60)
    
    if 'price' not in df.columns:
        print("⚠ Warning: No price column found")
        return df, ["No price column to clean"]
    
    original_sample = df['price'].head(5).tolist()
    original_dtype = str(df['price'].dtype)
    
    # Convert to string first
    df['price'] = df['price'].astype(str)
    
    # Remove common non-numeric characters
    df['price'] = df['price'].str.replace('$', '', regex=False)
    df['price'] = df['price'].str.replace(',', '', regex=False)
    df['price'] = df['price'].str.replace('+', '', regex=False)
    df['price'] = df['price'].str.replace(' ', '', regex=False)
    df['price'] = df['price'].str.replace('cad', '', case=False, regex=False)
    df['price'] = df['price'].str.replace('c\$', '', case=False, regex=False)
    
    # Remove text indicators
    text_indicators = ['call', 'ask', 'negotiable', 'contact', 'please', 'inquiry', 
                      'available', 'upon', 'request', 'email', 'phone']
    for indicator in text_indicators:
        df['price'] = df['price'].str.replace(indicator, '', case=False, regex=False)
    
    # Extract numeric values from strings like "1000-1200" or "1000+"
    def extract_price(value):
        if pd.isna(value):
            return np.nan
        
        # Find all numbers in the string
        numbers = re.findall(r'\d+\.?\d*', str(value))
        if numbers:
            # Take the first number (or average if range)
            if len(numbers) == 2 and '-' in str(value):
                return (float(numbers[0]) + float(numbers[1])) / 2
            return float(numbers[0])
        return np.nan
    
    df['price'] = df['price'].apply(extract_price)
    
    # Convert to numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Log statistics
    invalid_prices = df['price'].isna().sum()
    negative_prices = (df['price'] <= 0).sum()
    
    print(f"✓ Price column converted to numeric")
    print(f"  Invalid prices (NaN): {invalid_prices:,}")
    print(f"  Negative/zero prices: {negative_prices:,}")
    print(f"  Valid prices: {df['price'].notna().sum():,}")
    print(f"  Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    
    log_details = [
        f"Original dtype: {original_dtype}",
        f"Original samples: {original_sample}",
        f"Cleaned dtype: {df['price'].dtype}",
        f"Invalid prices removed: {invalid_prices:,}",
        f"Negative/zero prices: {negative_prices:,}",
        f"Valid prices remaining: {df['price'].notna().sum():,}",
        f"New price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}"
    ]
    
    return df, log_details


def clean_numeric_columns(df):
    """Clean beds, baths, and sq_feet columns"""
    print("\n" + "="*60)
    print("CLEANING NUMERIC COLUMNS")
    print("="*60)
    
    numeric_columns = ['beds', 'baths', 'sq_feet']
    log_details = []
    
    for col in numeric_columns:
        if col in df.columns:
            original_dtype = str(df[col].dtype)
            original_nulls = df[col].isna().sum()
            
            # Convert to string first to handle mixed types
            df[col] = df[col].astype(str)
            
            # Remove non-numeric characters
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = df[col].str.replace('+', '', regex=False)
            df[col] = df[col].str.replace('sqft', '', case=False, regex=False)
            df[col] = df[col].str.replace('sq ft', '', case=False, regex=False)
            df[col] = df[col].str.replace('ft²', '', regex=False)
            df[col] = df[col].str.replace(' ', '', regex=False)
            
            # Extract first number from ranges (e.g., "1-2" -> 1.5)
            def extract_numeric(value):
                if pd.isna(value) or value == 'nan':
                    return np.nan
                
                numbers = re.findall(r'\d+\.?\d*', str(value))
                if numbers:
                    if len(numbers) == 2 and '-' in str(value):
                        return (float(numbers[0]) + float(numbers[1])) / 2
                    return float(numbers[0])
                return np.nan
            
            df[col] = df[col].apply(extract_numeric)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            new_nulls = df[col].isna().sum()
            added_nulls = new_nulls - original_nulls
            
            print(f"✓ Cleaned: {col}")
            print(f"  Original dtype: {original_dtype}")
            print(f"  New dtype: {df[col].dtype}")
            print(f"  Added nulls from cleaning: {added_nulls:,}")
            
            log_details.append(f"\n{col.upper()}:")
            log_details.append(f"  Original dtype: {original_dtype}")
            log_details.append(f"  New dtype: {df[col].dtype}")
            log_details.append(f"  Added nulls: {added_nulls:,}")
            log_details.append(f"  Total nulls: {new_nulls:,}")
            
            if col == 'beds':
                bed_dist = df['beds'].value_counts().sort_index().head(10)
                log_details.append(f"  Most common values: {dict(bed_dist.head(5))}")
    
    return df, log_details
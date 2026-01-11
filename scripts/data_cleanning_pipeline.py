"""
COMPREHENSIVE CANADIAN RENTAL DATA CLEANING PIPELINE
Author: Data Analyst
Date: 2024

Cleans raw rental data from RentFaster Canada (25,000+ listings)
Prepares data for province-level analysis and dashboard creation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIGURATION
# ================================
RAW_DATA_PATH = Path('/Users/mastp/Documents/DataProject/rental/data/raw/rentals.csv')
CLEAN_DATA_PATH = Path('/Users/mastp/Documents/DataProject/rental/data/processed/rentals_clean.csv')
LOG_PATH = Path('/Users/mastp/Documents/DataProject/rental/logs/data_cleaning_log.txt')

# Create directories if they don't exist
CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ================================
# HELPER FUNCTIONS
# ================================
def initialize_logging():
    """Initialize logging system"""
    log_content = []
    log_content.append("="*70)
    log_content.append("CANADIAN RENTAL DATA CLEANING LOG")
    log_content.append(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append("="*70 + "\n")
    return log_content

def log_step(log_content, step_num, step_name, details=""):
    """Add a step to the log"""
    log_content.append(f"\nSTEP {step_num}: {step_name}")
    log_content.append("-"*40)
    if details:
        if isinstance(details, list):
            log_content.extend(details)
        else:
            log_content.append(details)
    return log_content

def save_log(log_content):
    """Save log to file"""
    with open(LOG_PATH, 'w') as f:
        f.write("\n".join(log_content))
    print(f"✓ Cleaning log saved to: {LOG_PATH}")

# ================================
# DATA CLEANING FUNCTIONS
# ================================
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

def remove_unnecessary_columns(df):
    """Remove columns that won't be used in analysis"""
    print("\n" + "="*60)
    print("REMOVING UNNECESSARY COLUMNS")
    print("="*60)
    
    columns_before = set(df.columns)
    
    # Columns to keep for analysis
    columns_to_keep = [
        'rentfaster_id', 'city', 'province', 'address', 
        'latitude', 'longitude', 'lease_term', 'type',
        'price', 'beds', 'baths', 'sq_feet', 'link',
        'furnishing', 'smoking', 'cats', 'dogs'
    ]
    
    # Remove availability_date as requested
    if 'availability_date' in df.columns:
        df = df.drop(columns=['availability_date'])
        print("✓ Removed: availability_date")
    
    # Ensure we have all required columns
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        print(f"⚠ Warning: Missing expected columns: {missing_cols}")
    
    columns_after = set(df.columns)
    removed = columns_before - columns_after
    
    log_details = [
        f"Columns removed: {', '.join(removed) if removed else 'None'}",
        f"Remaining columns: {len(df.columns)}",
        f"Columns kept: {', '.join(df.columns.tolist())}"
    ]
    
    return df, log_details

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

def handle_missing_values(df):
    """Handle missing values strategically"""
    print("\n" + "="*60)
    print("HANDLING MISSING VALUES")
    print("="*60)
    
    initial_rows = len(df)
    missing_report = []
    
    # Calculate missing values before cleaning
    missing_before = df.isna().sum()
    missing_pct_before = (missing_before / len(df) * 100).round(2)
    
    print("\nMISSING VALUES BEFORE CLEANING:")
    for col in df.columns:
        if missing_before[col] > 0:
            print(f"  {col}: {missing_before[col]:,} ({missing_pct_before[col]}%)")
            missing_report.append(f"{col}: {missing_before[col]:,} ({missing_pct_before[col]}%)")
    
    # Strategic imputation
    print("\nSTRATEGIC IMPUTATION:")
    
    # For beds: if missing but we have type info, make educated guess
    if 'beds' in df.columns and 'type' in df.columns:
        beds_missing = df['beds'].isna().sum()
        
        # Estimate beds based on property type
        type_to_beds = {
            'studio': 0,
            'apartment': 1,
            'condo': 2,
            'house': 3,
            'townhouse': 2,
            'duplex': 2,
            'basement': 1
        }
        
        def estimate_beds(row):
            if pd.isna(row['beds']) and pd.notna(row['type']):
                for prop_type, bed_count in type_to_beds.items():
                    if prop_type in str(row['type']).lower():
                        return bed_count
            return row['beds']
        
        df['beds'] = df.apply(estimate_beds, axis=1)
        beds_imputed = beds_missing - df['beds'].isna().sum()
        print(f"  Estimated beds for {beds_imputed:,} listings based on property type")
    
    # For baths: if missing but we have beds, estimate
    if 'baths' in df.columns and 'beds' in df.columns:
        baths_missing = df['baths'].isna().sum()
        
        # Simple rule: usually similar to number of beds
        df['baths'] = df.apply(
            lambda x: x['beds'] if pd.isna(x['baths']) and pd.notna(x['beds']) else x['baths'],
            axis=1
        )
        baths_imputed = baths_missing - df['baths'].isna().sum()
        print(f"  Estimated baths for {baths_imputed:,} listings based on beds")
    
    # For sq_feet: if missing, we'll keep as NaN (can't reliably estimate)
    if 'sq_feet' in df.columns:
        sqft_missing = df['sq_feet'].isna().sum()
        print(f"  Square footage missing for {sqft_missing:,} listings (kept as NaN)")
    
    # Remove rows with critical missing data
    print("\nREMOVING ROWS WITH CRITICAL MISSING DATA:")
    critical_cols = ['price', 'province', 'city']
    rows_before = len(df)
    
    df = df.dropna(subset=critical_cols)
    rows_removed = rows_before - len(df)
    
    print(f"  Removed {rows_removed:,} rows missing price, province, or city")
    
    # Calculate missing values after cleaning
    missing_after = df.isna().sum()
    missing_pct_after = (missing_after / len(df) * 100).round(2)
    
    print("\nMISSING VALUES AFTER CLEANING:")
    for col in df.columns:
        if missing_after[col] > 0:
            print(f"  {col}: {missing_after[col]:,} ({missing_pct_after[col]}%)")
    
    log_details = [
        f"Initial rows: {initial_rows:,}",
        f"Rows removed (critical missing data): {rows_removed:,}",
        f"Final rows: {len(df):,}",
        "\nMissing values before cleaning:",
        *[f"  {item}" for item in missing_report],
        f"\nMissing values after cleaning:",
        *[f"  {col}: {missing_after[col]:,} ({missing_pct_after[col]}%)" 
          for col in df.columns if missing_after[col] > 0]
    ]
    
    return df, log_details

def remove_unrealistic_outliers(df):
    """Remove unrealistic outliers while keeping legitimate data"""
    print("\n" + "="*60)
    print("REMOVING UNREALISTIC OUTLIERS")
    print("="*60)
    
    initial_rows = len(df)
    
    # Price-based outlier removal
    if 'price' in df.columns:
        # Calculate IQR
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Reasonable bounds (3x IQR is less aggressive than 1.5x)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Also set absolute minimums/maximums based on Canadian rental market reality
        absolute_min = 300  # Minimum realistic monthly rent
        absolute_max = 30000  # Maximum realistic monthly rent
        
        lower_bound = max(lower_bound, absolute_min)
        upper_bound = min(upper_bound, absolute_max)
        
        # Identify outliers
        price_outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
        print(f"Price outliers identified: {len(price_outliers):,}")
        print(f"  Lower bound: ${lower_bound:,.0f}")
        print(f"  Upper bound: ${upper_bound:,.0f}")
        print(f"  Outlier range: ${price_outliers['price'].min():,.0f} - ${price_outliers['price'].max():,.0f}")
        
        # Remove price outliers
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    
    # Size-based outlier removal (sq_feet)
    if 'sq_feet' in df.columns:
        # Remove unrealistic sizes (too small or too large for residential)
        size_outliers = df[(df['sq_feet'] < 100) | (df['sq_feet'] > 10000)]
        if len(size_outliers) > 0:
            print(f"Size outliers identified: {len(size_outliers):,}")
            print(f"  Outlier range: {size_outliers['sq_feet'].min():,.0f} - {size_outliers['sq_feet'].max():,.0f} sq ft")
            df = df[(df['sq_feet'] >= 100) & (df['sq_feet'] <= 10000)]
    
    # Bedroom outlier removal
    if 'beds' in df.columns:
        # Remove listings with unrealistic bedroom counts
        bed_outliers = df[df['beds'] > 10]
        if len(bed_outliers) > 0:
            print(f"Bedroom outliers identified: {len(bed_outliers):,}")
            print(f"  Max bedrooms in outliers: {bed_outliers['beds'].max():.0f}")
            df = df[df['beds'] <= 10]
    
    rows_removed = initial_rows - len(df)
    print(f"\nTotal outliers removed: {rows_removed:,}")
    print(f"Data remaining: {len(df):,} rows ({len(df)/initial_rows*100:.1f}% of original)")
    
    log_details = [
        f"Initial rows: {initial_rows:,}",
        f"Rows removed as outliers: {rows_removed:,}",
        f"Final rows: {len(df):,}",
        f"Percentage kept: {len(df)/initial_rows*100:.1f}%"
    ]
    
    return df, log_details

def validate_geographic_data(df):
    """Validate and clean geographic data"""
    print("\n" + "="*60)
    print("VALIDATING GEOGRAPHIC DATA")
    print("="*60)
    
    log_details = []
    
    # Province validation
    if 'province' in df.columns:
        valid_provinces = [
            'alberta', 'british columbia', 'manitoba', 'new brunswick',
            'newfoundland and labrador', 'nova scotia', 'ontario',
            'prince edward island', 'quebec', 'saskatchewan',
            'northwest territories', 'nunavut', 'yukon'
        ]
        
        invalid_provinces = df[~df['province'].isin(valid_provinces)]['province'].unique()
        if len(invalid_provinces) > 0:
            print(f"Invalid province names found: {invalid_provinces}")
            
            # Try to fix common issues
            province_fixes = {
                'bc': 'british columbia',
                'ontario (toronto)': 'ontario',
                'quebec (montreal)': 'quebec',
                'alb': 'alberta',
                'man': 'manitoba',
                'sask': 'saskatchewan'
            }
            
            df['province'] = df['province'].replace(province_fixes)
            
            # Remove any remaining invalid provinces
            rows_before = len(df)
            df = df[df['province'].isin(valid_provinces)]
            rows_removed = rows_before - len(df)
            
            print(f"Removed {rows_removed:,} rows with invalid provinces")
            log_details.append(f"Rows removed (invalid provinces): {rows_removed:,}")
    
    # City cleaning
    if 'city' in df.columns:
        # Remove numbers and special characters from city names
        df['city'] = df['city'].str.replace(r'\d+', '', regex=True)
        df['city'] = df['city'].str.replace(r'[^\w\s]', '', regex=True)
        df['city'] = df['city'].str.strip()
        
        # Title case for readability
        df['city'] = df['city'].str.title()
        
        print(f"Cleaned city names")
        log_details.append("City names cleaned and formatted")
    
    # Coordinate validation
    if all(col in df.columns for col in ['latitude', 'longitude']):
        # Check for valid Canadian coordinates
        valid_coords = df[
            (df['latitude'].between(41.7, 83.1)) &  # Canada's latitude range
            (df['longitude'].between(-141.0, -52.6))  # Canada's longitude range
        ]
        
        invalid_coords = len(df) - len(valid_coords)
        if invalid_coords > 0:
            print(f"Found {invalid_coords:,} rows with coordinates outside Canada")
            log_details.append(f"Rows with non-Canadian coordinates: {invalid_coords:,}")
    
    # Count provinces and cities
    province_count = df['province'].nunique() if 'province' in df.columns else 0
    city_count = df['city'].nunique() if 'city' in df.columns else 0
    
    print(f"\nGeographic Summary:")
    print(f"  Provinces covered: {province_count}")
    print(f"  Cities covered: {city_count}")
    
    if province_count > 0:
        province_dist = df['province'].value_counts().head()
        print(f"  Top provinces: {dict(province_dist)}")
        log_details.append(f"Top provinces by listings: {dict(province_dist)}")
    
    log_details.append(f"Total provinces: {province_count}")
    log_details.append(f"Total cities: {city_count}")
    
    return df, log_details

def final_data_quality_check(df):
    """Perform final data quality assessment"""
    print("\n" + "="*60)
    print("FINAL DATA QUALITY CHECK")
    print("="*60)
    
    log_details = []
    
    # Basic statistics
    total_rows = len(df)
    total_columns = len(df.columns)
    
    print(f"Dataset Shape: {total_rows:,} rows × {total_columns} columns")
    log_details.append(f"Final shape: {total_rows:,} rows × {total_columns} columns")
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Duplicate rows found: {duplicate_count:,}")
        df = df.drop_duplicates()
        print(f"Removed duplicates, new shape: {len(df):,} rows")
        log_details.append(f"Duplicates removed: {duplicate_count:,}")
    
    # Data type summary
    print("\nDATA TYPES:")
    dtype_summary = df.dtypes.value_counts().to_dict()
    for dtype, count in dtype_summary.items():
        print(f"  {dtype}: {count} columns")
        log_details.append(f"Data type {dtype}: {count} columns")
    
    # Missing values final check
    missing_final = df.isna().sum()
    missing_cols = missing_final[missing_final > 0]
    
    if len(missing_cols) > 0:
        print("\nREMAINING MISSING VALUES:")
        for col, count in missing_cols.items():
            pct = (count / total_rows * 100)
            print(f"  {col}: {count:,} ({pct:.1f}%)")
            log_details.append(f"Missing in {col}: {count:,} ({pct:.1f}%)")
    else:
        print("\n✓ No missing values in any column")
        log_details.append("No missing values in any column")
    
    # Key column statistics
    print("\nKEY STATISTICS:")
    if 'price' in df.columns:
        price_stats = {
            'min': df['price'].min(),
            'max': df['price'].max(),
            'mean': df['price'].mean(),
            'median': df['price'].median(),
            'std': df['price'].std()
        }
        print(f"Price: ${price_stats['mean']:,.0f} avg, ${price_stats['median']:,.0f} median")
        print(f"       ${price_stats['min']:,.0f} min, ${price_stats['max']:,.0f} max")
        log_details.append(f"Price stats: Avg=${price_stats['mean']:,.0f}, Med=${price_stats['median']:,.0f}")
    
    if 'beds' in df.columns:
        bed_stats = df['beds'].describe()
        print(f"Bedrooms: {bed_stats['mean']:.1f} avg, {bed_stats['50%']:.0f} median")
        log_details.append(f"Bedrooms: Avg={bed_stats['mean']:.1f}, Med={bed_stats['50%']:.0f}")
    
    # Province distribution
    if 'province' in df.columns:
        province_dist = df['province'].value_counts()
        print(f"\nPROVINCE DISTRIBUTION:")
        print(f"  Total provinces: {len(province_dist)}")
        print(f"  Top 5 provinces:")
        for province, count in province_dist.head().items():
            pct = (count / total_rows * 100)
            print(f"    {province.title()}: {count:,} ({pct:.1f}%)")
            log_details.append(f"{province.title()}: {count:,} ({pct:.1f}%)")
    
    return df, log_details

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

# ================================
# MAIN EXECUTION
# ================================
def main():
    """Main data cleaning pipeline"""
    print("\n" + "="*60)
    print("CANADIAN RENTAL DATA CLEANING PIPELINE")
    print("="*60)
    print("Processing 25,000+ rental listings from RentFaster Canada\n")
    
    # Initialize logging
    log_content = initialize_logging()
    
    try:
        # Step 1: Load raw data
        df, step1_log = load_raw_data(RAW_DATA_PATH)
        log_content = log_step(log_content, 1, "LOAD RAW DATA", step1_log)
        
        # Step 2: Remove unnecessary columns
        df, step2_log = remove_unnecessary_columns(df)
        log_content = log_step(log_content, 2, "REMOVE UNNECESSARY COLUMNS", step2_log)
        
        # Step 3: Normalize text columns
        df, step3_log = normalize_text_columns(df)
        log_content = log_step(log_content, 3, "NORMALIZE TEXT COLUMNS", step3_log)
        
        # Step 4: Clean price column
        df, step4_log = clean_price_column(df)
        log_content = log_step(log_content, 4, "CLEAN PRICE COLUMN", step4_log)
        
        # Step 5: Clean numeric columns
        df, step5_log = clean_numeric_columns(df)
        log_content = log_step(log_content, 5, "CLEAN NUMERIC COLUMNS", step5_log)
        
        # Step 6: Clean pet columns
        df, step6_log = clean_pet_columns(df)
        log_content = log_step(log_content, 6, "CLEAN PET COLUMNS", step6_log)
        
        # Step 7: Handle missing values
        df, step7_log = handle_missing_values(df)
        log_content = log_step(log_content, 7, "HANDLE MISSING VALUES", step7_log)
        
        # Step 8: Remove unrealistic outliers
        df, step8_log = remove_unrealistic_outliers(df)
        log_content = log_step(log_content, 8, "REMOVE UNREALISTIC OUTLIERS", step8_log)
        
        # Step 9: Validate geographic data
        df, step9_log = validate_geographic_data(df)
        log_content = log_step(log_content, 9, "VALIDATE GEOGRAPHIC DATA", step9_log)
        
        # Step 10: Final data quality check
        df, step10_log = final_data_quality_check(df)
        log_content = log_step(log_content, 10, "FINAL DATA QUALITY CHECK", step10_log)
        
        # Step 11: Save cleaned data
        save_log_details = save_clean_data(df, CLEAN_DATA_PATH)
        log_content = log_step(log_content, 11, "SAVE CLEANED DATA", save_log_details)
        
        # Add final summary to log
        log_content.append("\n" + "="*70)
        log_content.append("CLEANING PIPELINE COMPLETE - SUMMARY")
        log_content.append("="*70)
        log_content.append(f"Total processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append(f"Final dataset: {len(df):,} rows × {len(df.columns)} columns")
        log_content.append(f"Output file: {CLEAN_DATA_PATH}")
        
        # Save log
        save_log(log_content)
        
        # Final output
        print("\n" + "="*60)
        print("CLEANING PIPELINE COMPLETE!")
        print("="*60)
        print(f"\n📊 FINAL DATASET SUMMARY:")
        print(f"  • Rows: {len(df):,}")
        print(f"  • Columns: {len(df.columns)}")
        print(f"  • File: {CLEAN_DATA_PATH}")
        
        print(f"\n🔧 COLUMNS AVAILABLE FOR ANALYSIS:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2}. {col}")
        
        print(f"\n📈 KEY STATISTICS:")
        if 'price' in df.columns:
            print(f"  • Average rent: ${df['price'].mean():,.0f}")
            print(f"  • Median rent: ${df['price'].median():,.0f}")
            print(f"  • Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        
        if 'province' in df.columns:
            print(f"  • Provinces covered: {df['province'].nunique()}")
            print(f"  • Top province: {df['province'].value_counts().index[0].title()}")
        
        if 'beds' in df.columns:
            print(f"  • Average bedrooms: {df['beds'].mean():.1f}")
        
        print(f"\n✅ NEXT STEPS:")
        print(f"  1. Run province_analysis.py on the cleaned data")
        print(f"  2. Review cleaning log: {LOG_PATH}")
        print(f"  3. Create dashboard using the clean dataset")
        
        print(f"\n📋 CLEANING LOG SAVED TO: {LOG_PATH}")
        
        return df
        
    except Exception as e:
        print(f"\n❌ ERROR IN CLEANING PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error to log
        log_content.append(f"\n❌ ERROR: {e}")
        log_content.append(traceback.format_exc())
        save_log(log_content)
        
        raise

# ================================
# EXECUTION
# ================================
if __name__ == "__main__":
    cleaned_data = main()
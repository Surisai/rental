"""
Data validation and quality check utilities.
"""

import pandas as pd
import numpy as np


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
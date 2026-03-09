"""
Province analysis utilities.
"""

import pandas as pd
import numpy as np


def analyze_by_province(df):
    """Generate comprehensive analysis for each province"""
    print("\n" + "="*60)
    print("STEP 3: PROVINCE-SPECIFIC ANALYSIS")
    print("="*60)
    
    if 'province' not in df.columns:
        print("✗ No 'province' column found in data")
        return {}
    
    provinces = df['province'].unique()
    print(f"Found {len(provinces)} provinces in data:")
    print("-"*40)
    
    province_results = {}
    
    for province in sorted(provinces):
        print(f"\nAnalyzing: {province.title()}")
        
        # Filter data for this province
        province_data = df[df['province'] == province].copy()
        
        if len(province_data) < 5:  # Skip provinces with too little data
            print(f"  ⚠ Skipping - only {len(province_data)} listings")
            continue
        
        # Calculate REAL ranges from actual data
        price_stats = {
            'min_price': province_data['price'].min(),
            'max_price': province_data['price'].max(),
            'mean_price': province_data['price'].mean(),
            'median_price': province_data['price'].median(),
            'std_price': province_data['price'].std(),
            'q1_price': province_data['price'].quantile(0.25),
            'q3_price': province_data['price'].quantile(0.75),
            'iqr_price': province_data['price'].quantile(0.75) - province_data['price'].quantile(0.25)
        }
        
        # Identify potential outliers using IQR method
        q1 = price_stats['q1_price']
        q3 = price_stats['q3_price']
        iqr = price_stats['iqr_price']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Calculate outliers
        outliers = province_data[(province_data['price'] < lower_bound) | (province_data['price'] > upper_bound)]
        price_stats['outlier_count'] = len(outliers)
        price_stats['outlier_percentage'] = (len(outliers) / len(province_data)) * 100
        
        # Calculate reasonable range (excluding outliers)
        reasonable_data = province_data[(province_data['price'] >= lower_bound) & (province_data['price'] <= upper_bound)]
        price_stats['reasonable_min'] = reasonable_data['price'].min() if len(reasonable_data) > 0 else price_stats['min_price']
        price_stats['reasonable_max'] = reasonable_data['price'].max() if len(reasonable_data) > 0 else price_stats['max_price']
        
        # Property type distribution
        if 'type' in province_data.columns:
            type_dist = province_data['type'].value_counts().head(10).to_dict()
            price_stats['most_common_type'] = province_data['type'].mode()[0] if not province_data['type'].mode().empty else 'N/A'
        
        # Bedroom distribution
        if 'beds' in province_data.columns:
            beds_dist = province_data['beds'].value_counts().sort_index().head(10).to_dict()
            price_stats['most_common_beds'] = province_data['beds'].mode()[0] if not province_data['beds'].mode().empty else 'N/A'
            price_stats['avg_beds'] = province_data['beds'].mean()
        
        # City analysis within province
        if 'city' in province_data.columns:
            city_stats = province_data.groupby('city').agg(
                listings=('price', 'count'),
                avg_price=('price', 'mean'),
                min_price=('price', 'min'),
                max_price=('price', 'max')
            ).sort_values('avg_price', ascending=False)
            
            price_stats['top_city'] = city_stats.index[0] if len(city_stats) > 0 else 'N/A'
            price_stats['top_city_avg_price'] = city_stats.iloc[0]['avg_price'] if len(city_stats) > 0 else 0
            price_stats['city_count'] = len(city_stats)
        
        # Size analysis
        if 'sq_feet' in province_data.columns:
            size_stats = province_data['sq_feet'].dropna()
            if len(size_stats) > 0:
                price_stats['avg_size'] = size_stats.mean()
                price_stats['min_size'] = size_stats.min()
                price_stats['max_size'] = size_stats.max()
        
        # Pets analysis
        if 'cats' in province_data.columns and 'dogs' in province_data.columns:
            cats_allowed = province_data['cats'].value_counts(normalize=True).get('yes', 0) * 100
            dogs_allowed = province_data['dogs'].value_counts(normalize=True).get('yes', 0) * 100
            price_stats['cats_allowed_pct'] = cats_allowed
            price_stats['dogs_allowed_pct'] = dogs_allowed
        
        # Add basic counts
        price_stats['total_listings'] = len(province_data)
        
        # Store results
        province_results[province] = {
            'stats': price_stats,
            'data': province_data,
            'city_stats': city_stats if 'city' in province_data.columns else None,
            'outliers': outliers
        }
        
        # Print summary
        print(f"  ✓ Listings: {price_stats['total_listings']:,}")
        print(f"  ✓ Price Range: ${price_stats['min_price']:,.0f} - ${price_stats['max_price']:,.0f}")
        print(f"  ✓ Average Price: ${price_stats['mean_price']:,.0f}")
        print(f"  ✓ Reasonable Range: ${price_stats['reasonable_min']:,.0f} - ${price_stats['reasonable_max']:,.0f}")
        print(f"  ✓ Outliers: {price_stats['outlier_count']} ({price_stats['outlier_percentage']:.1f}%)")
    
    return province_results
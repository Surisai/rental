"""
Data export utilities for province analysis.
"""

import pandas as pd
from .config import OUTPUT_DIRS


def export_province_data(province_results):
    """Export detailed CSV files for each province"""
    print("\n" + "="*60)
    print("STEP 4: EXPORTING PROVINCE-SPECIFIC DATA")
    print("="*60)
    
    # 4.1 Export individual province data
    print("\nExporting individual province data to: provinces/")
    for province, results in province_results.items():
        if results['data'] is not None and len(results['data']) > 0:
            # Clean province name for filename
            clean_name = province.replace(' ', '_').replace('/', '_').lower()
            
            # Export full province data
            province_file = OUTPUT_DIRS['provinces'] / f"{clean_name}_rentals.csv"
            results['data'].to_csv(province_file, index=False)
            
            # Export city summary for this province
            if results['city_stats'] is not None:
                city_file = OUTPUT_DIRS['provinces'] / f"{clean_name}_city_summary.csv"
                results['city_stats'].to_csv(city_file)
    
    print(f"✓ Individual province data exported to: {OUTPUT_DIRS['provinces']}")
    
    # 4.2 Create master province comparison file
    print("\nCreating master province comparison...")
    province_comparison = []
    
    for province, results in province_results.items():
        stats = results['stats']
        
        comparison = {
            'province': province.title(),
            'total_listings': stats['total_listings'],
            'min_price': stats['min_price'],
            'max_price': stats['max_price'],
            'mean_price': stats['mean_price'],
            'median_price': stats['median_price'],
            'reasonable_min': stats.get('reasonable_min', stats['min_price']),
            'reasonable_max': stats.get('reasonable_max', stats['max_price']),
            'price_std': stats['std_price'],
            'outlier_count': stats['outlier_count'],
            'outlier_percentage': stats['outlier_percentage'],
            'city_count': stats.get('city_count', 0),
            'avg_beds': stats.get('avg_beds', 0),
            'avg_size': stats.get('avg_size', 0),
            'cats_allowed_pct': stats.get('cats_allowed_pct', 0),
            'dogs_allowed_pct': stats.get('dogs_allowed_pct', 0)
        }
        province_comparison.append(comparison)
    
    # Create DataFrame and sort
    comparison_df = pd.DataFrame(province_comparison)
    comparison_df = comparison_df.sort_values('mean_price', ascending=False)
    
    # Export comparison to data folder
    comparison_file = OUTPUT_DIRS['data'] / 'province_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✓ Province comparison exported: {comparison_file}")
    
    # 4.3 Create detailed stats file
    print("\nCreating detailed province statistics...")
    detailed_stats = []
    
    for province, results in province_results.items():
        stats = results['stats']
        
        detailed = {
            'province': province.title(),
            'listings': stats['total_listings'],
            # Price Statistics
            'absolute_min': stats['min_price'],
            'absolute_max': stats['max_price'],
            'mean': stats['mean_price'],
            'median': stats['median_price'],
            'std_dev': stats['std_price'],
            'q1': stats['q1_price'],
            'q3': stats['q3_price'],
            'iqr': stats['iqr_price'],
            # Calculated Ranges
            'lower_bound': stats['q1_price'] - 1.5 * stats['iqr_price'],
            'upper_bound': stats['q3_price'] + 1.5 * stats['iqr_price'],
            'reasonable_min': stats.get('reasonable_min', stats['min_price']),
            'reasonable_max': stats.get('reasonable_max', stats['max_price']),
            # Outlier Info
            'outliers': stats['outlier_count'],
            'outlier_pct': stats['outlier_percentage'],
            # Market Characteristics
            'most_common_type': stats.get('most_common_type', 'N/A'),
            'most_common_beds': stats.get('most_common_beds', 'N/A'),
            'top_city': stats.get('top_city', 'N/A'),
            'top_city_price': stats.get('top_city_avg_price', 0)
        }
        detailed_stats.append(detailed)
    
    detailed_df = pd.DataFrame(detailed_stats)
    detailed_file = OUTPUT_DIRS['data'] / 'province_detailed_stats.csv'
    detailed_df.to_csv(detailed_file, index=False)
    print(f"✓ Detailed statistics exported: {detailed_file}")
    
    return comparison_df, detailed_df
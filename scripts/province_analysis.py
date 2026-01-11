"""
Rental Housing Canada Analysis - Province-Specific Analysis
WITH ENHANCED FOLDER ORGANIZATION
Author: Data Analyst
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIGURATION WITH FOLDER STRUCTURE
# ================================
INPUT_PATH = Path('/Users/mastp/Documents/DataProject/rental/data/processed/rentals_clean.csv')
# Main output directory
MAIN_OUTPUT_DIR = Path('/Users/mastp/Documents/DataProject/rental/outputs/province_analysis')

# Create structured subdirectories
OUTPUT_DIRS = {
    'main': MAIN_OUTPUT_DIR,
    'reports': MAIN_OUTPUT_DIR / 'reports',
    'data': MAIN_OUTPUT_DIR / 'data',
    'visualizations': MAIN_OUTPUT_DIR / 'visualizations',
    'provinces': MAIN_OUTPUT_DIR / 'data' / 'provinces',
    'province_charts': MAIN_OUTPUT_DIR / 'visualizations' / 'province_distributions',
    'summary_charts': MAIN_OUTPUT_DIR / 'visualizations' / 'summary_charts'
}

# Create all directories
for dir_name, dir_path in OUTPUT_DIRS.items():
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {dir_path}")

# ================================
# STEP 1: LOAD DATA
# ================================
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

# ================================
# STEP 2: DATA PREPARATION
# ================================
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

# ================================
# STEP 3: PROVINCE-SPECIFIC ANALYSIS
# ================================
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

# ================================
# STEP 4: CREATE PROVINCE CSV FILES
# ================================
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

# ================================
# STEP 5: PROVINCE VISUALIZATIONS
# ================================
def create_province_visualizations(comparison_df, province_results):
    """Create visualizations comparing provinces"""
    print("\n" + "="*60)
    print("STEP 5: PROVINCE VISUALIZATIONS")
    print("="*60)
    
    if comparison_df.empty:
        print("No comparison data available")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Province-Level Rental Market Analysis', fontsize=18, fontweight='bold')
    
    # Plot 1: Average Price by Province
    axes[0, 0].barh(comparison_df['province'], comparison_df['mean_price'], color='steelblue')
    axes[0, 0].set_title('1. Average Rental Price by Province', fontweight='bold')
    axes[0, 0].set_xlabel('Average Price ($)')
    axes[0, 0].invert_yaxis()
    for i, v in enumerate(comparison_df['mean_price']):
        axes[0, 0].text(v + 20, i, f'${v:,.0f}', va='center')
    
    # Plot 2: Price Range by Province (Min, Mean, Max)
    provinces = comparison_df['province']
    mins = comparison_df['reasonable_min']
    means = comparison_df['mean_price']
    maxs = comparison_df['reasonable_max']
    
    x = range(len(provinces))
    axes[0, 1].errorbar(x, means, yerr=[means - mins, maxs - means], 
                       fmt='o', color='coral', ecolor='lightcoral', 
                       elinewidth=3, capsize=5, markersize=8)
    axes[0, 1].set_title('2. Price Ranges by Province', fontweight='bold')
    axes[0, 1].set_xlabel('Province')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(provinces, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Number of Listings by Province
    axes[1, 0].bar(provinces, comparison_df['total_listings'], color='seagreen')
    axes[1, 0].set_title('3. Number of Listings by Province', fontweight='bold')
    axes[1, 0].set_xlabel('Province')
    axes[1, 0].set_ylabel('Number of Listings')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(comparison_df['total_listings']):
        axes[1, 0].text(i, v + 5, f'{v:,}', ha='center')
    
    # Plot 4: Pet Friendliness by Province
    if 'cats_allowed_pct' in comparison_df.columns:
        x = range(len(provinces))
        width = 0.35
        axes[1, 1].bar(x, comparison_df['cats_allowed_pct'], width, label='Cats Allowed', color='gold')
        axes[1, 1].bar([i + width for i in x], comparison_df['dogs_allowed_pct'], width, 
                      label='Dogs Allowed', color='orange')
        axes[1, 1].set_title('4. Pet-Friendly Listings by Province', fontweight='bold')
        axes[1, 1].set_xlabel('Province')
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].set_xticks([i + width/2 for i in x])
        axes[1, 1].set_xticklabels(provinces, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save to summary_charts folder
    summary_chart_file = OUTPUT_DIRS['summary_charts'] / 'province_analysis_summary.png'
    plt.savefig(summary_chart_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Province analysis chart saved: {summary_chart_file}")
    
    # Create individual province distribution plots
    print("\nCreating individual province distributions...")
    
    for province, results in province_results.items():
        if results['data'] is not None and len(results['data']) > 5:
            plt.figure(figsize=(10, 6))
            
            # Create histogram with KDE
            sns.histplot(results['data']['price'], bins=30, kde=True, color='purple')
            
            # Add statistics lines
            stats = results['stats']
            plt.axvline(stats['mean_price'], color='red', linestyle='--', 
                       label=f'Mean: ${stats["mean_price"]:,.0f}')
            plt.axvline(stats['median_price'], color='green', linestyle='--', 
                       label=f'Median: ${stats["median_price"]:,.0f}')
            plt.axvline(stats['reasonable_min'], color='blue', linestyle=':', 
                       label=f'Reasonable Min: ${stats["reasonable_min"]:,.0f}')
            plt.axvline(stats['reasonable_max'], color='blue', linestyle=':', 
                       label=f'Reasonable Max: ${stats["reasonable_max"]:,.0f}')
            
            plt.title(f'Price Distribution in {province.title()}\n'
                     f'{stats["total_listings"]:,} Listings | Range: ${stats["min_price"]:,.0f} - ${stats["max_price"]:,.0f}')
            plt.xlabel('Price ($)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save individual province plot to province_charts folder
            clean_name = province.replace(' ', '_').replace('/', '_').lower()
            distribution_file = OUTPUT_DIRS['province_charts'] / f"{clean_name}_distribution.png"
            plt.savefig(distribution_file, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Individual province distributions saved in: {OUTPUT_DIRS['province_charts']}")
    
    # Create additional visualizations
    create_additional_visualizations(comparison_df, province_results)

def create_additional_visualizations(comparison_df, province_results):
    """Create additional specialized visualizations"""
    print("\nCreating additional visualizations...")
    
    # 1. Price vs Listings Scatter Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(comparison_df['total_listings'], comparison_df['mean_price'], 
                         s=comparison_df['total_listings']/100, alpha=0.6,
                         c=range(len(comparison_df)), cmap='viridis')
    
    # Add province labels
    for i, row in comparison_df.iterrows():
        plt.annotate(row['province'], (row['total_listings'], row['mean_price']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Number of Listings')
    plt.ylabel('Average Price ($)')
    plt.title('Market Size vs Price by Province')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Province Rank')
    
    scatter_file = OUTPUT_DIRS['summary_charts'] / 'market_size_vs_price.png'
    plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Market size vs price scatter plot saved")
    
    # 2. Outlier Analysis Chart
    plt.figure(figsize=(12, 6))
    x = range(len(comparison_df))
    bars = plt.bar(x, comparison_df['outlier_percentage'], color='salmon')
    plt.xlabel('Province')
    plt.ylabel('Outlier Percentage (%)')
    plt.title('Percentage of Outliers by Province')
    plt.xticks(x, comparison_df['province'], rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars, comparison_df['outlier_percentage']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    outlier_file = OUTPUT_DIRS['summary_charts'] / 'outlier_analysis.png'
    plt.savefig(outlier_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Outlier analysis chart saved")

# ================================
# STEP 6: GENERATE ANALYSIS REPORT
# ================================
def generate_analysis_report(df, province_results, comparison_df):
    """Generate comprehensive analysis report"""
    print("\n" + "="*60)
    print("STEP 6: GENERATING ANALYSIS REPORT")
    print("="*60)
    
    report_path = OUTPUT_DIRS['reports'] / 'province_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RENTAL HOUSING CANADA - PROVINCE ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Listings Analyzed: {len(df):,}\n")
        f.write(f"Number of Provinces: {len(province_results)}\n")
        f.write(f"Data Source: {INPUT_PATH.name}\n")
        f.write("\n" + "-"*70 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL MARKET SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Overall Price Range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}\n")
        f.write(f"Overall Average Price: ${df['price'].mean():,.0f}\n")
        f.write(f"Overall Median Price: ${df['price'].median():,.0f}\n\n")
        
        # Province Rankings
        f.write("PROVINCE RANKINGS BY AVERAGE PRICE\n")
        f.write("-"*40 + "\n")
        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            f.write(f"{i}. {row['province']}: ${row['mean_price']:,.0f} "
                   f"(±${row['price_std']:,.0f}, {row['total_listings']:,} listings)\n")
        f.write("\n")
        
        # Detailed Province Analysis
        f.write("DETAILED PROVINCE ANALYSIS\n")
        f.write("="*40 + "\n\n")
        
        for province, results in province_results.items():
            stats = results['stats']
            
            f.write(f"{province.upper()}\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Listings: {stats['total_listings']:,}\n")
            f.write(f"Cities Covered: {stats.get('city_count', 'N/A')}\n")
            f.write(f"Absolute Price Range: ${stats['min_price']:,.0f} - ${stats['max_price']:,.0f}\n")
            f.write(f"Reasonable Price Range: ${stats.get('reasonable_min', stats['min_price']):,.0f} "
                   f"- ${stats.get('reasonable_max', stats['max_price']):,.0f}\n")
            f.write(f"Average Price: ${stats['mean_price']:,.0f}\n")
            f.write(f"Median Price: ${stats['median_price']:,.0f}\n")
            f.write(f"Price Standard Deviation: ${stats['std_price']:,.0f}\n")
            f.write(f"Outliers Identified: {stats['outlier_count']} ({stats['outlier_percentage']:.1f}%)\n")
            
            if 'most_common_type' in stats:
                f.write(f"Most Common Property Type: {stats['most_common_type']}\n")
            
            if 'avg_beds' in stats:
                f.write(f"Average Bedrooms: {stats['avg_beds']:.1f}\n")
            
            if 'cats_allowed_pct' in stats:
                f.write(f"Pet Friendly: Cats {stats['cats_allowed_pct']:.1f}% | "
                       f"Dogs {stats['dogs_allowed_pct']:.1f}%\n")
            
            f.write(f"Top City: {stats.get('top_city', 'N/A').title()} "
                   f"(${stats.get('top_city_avg_price', 0):,.0f})\n\n")
        
        # Market Insights
        f.write("MARKET INSIGHTS\n")
        f.write("="*40 + "\n\n")
        
        # Most expensive province
        most_expensive = comparison_df.iloc[0]
        f.write(f"• Most Expensive Province: {most_expensive['province']} "
               f"(Average: ${most_expensive['mean_price']:,.0f})\n")
        
        # Most affordable province
        most_affordable = comparison_df.iloc[-1]
        f.write(f"• Most Affordable Province: {most_affordable['province']} "
               f"(Average: ${most_affordable['mean_price']:,.0f})\n")
        
        # Highest variability
        highest_var = comparison_df.loc[comparison_df['price_std'].idxmax()]
        f.write(f"• Highest Price Variability: {highest_var['province']} "
               f"(Std Dev: ${highest_var['price_std']:,.0f})\n")
        
        # Most listings
        most_listings = comparison_df.loc[comparison_df['total_listings'].idxmax()]
        f.write(f"• Most Active Market: {most_listings['province']} "
               f"({most_listings['total_listings']:,} listings)\n")
        
        # Pet friendliest
        if 'cats_allowed_pct' in comparison_df.columns:
            pet_friendly = comparison_df.loc[comparison_df['cats_allowed_pct'].idxmax()]
            f.write(f"• Most Pet-Friendly: {pet_friendly['province']} "
                   f"({pet_friendly['cats_allowed_pct']:.1f}% allow cats)\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("FOLDER STRUCTURE\n")
        f.write("-"*70 + "\n")
        f.write(f"Main Output Directory: {MAIN_OUTPUT_DIR}\n")
        f.write(f"├── reports/\n")
        f.write(f"│   └── province_analysis_report.txt (this file)\n")
        f.write(f"├── data/\n")
        f.write(f"│   ├── province_comparison.csv\n")
        f.write(f"│   ├── province_detailed_stats.csv\n")
        f.write(f"│   └── provinces/\n")
        f.write(f"│       ├── alberta_rentals.csv\n")
        f.write(f"│       ├── alberta_city_summary.csv\n")
        f.write(f"│       └── [other province files...]\n")
        f.write(f"└── visualizations/\n")
        f.write(f"    ├── summary_charts/\n")
        f.write(f"    │   ├── province_analysis_summary.png\n")
        f.write(f"    │   ├── market_size_vs_price.png\n")
        f.write(f"    │   └── outlier_analysis.png\n")
        f.write(f"    └── province_distributions/\n")
        f.write(f"        ├── alberta_distribution.png\n")
        f.write(f"        └── [other province distributions...]\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("OUTPUT FILES GENERATED:\n")
        f.write("-"*70 + "\n")
        f.write("1. reports/province_analysis_report.txt - Comprehensive analysis report\n")
        f.write("2. data/province_comparison.csv - Summary comparison of all provinces\n")
        f.write("3. data/province_detailed_stats.csv - Detailed statistics for each province\n")
        f.write("4. data/provinces/ folder - Individual CSV files for each province\n")
        f.write("5. visualizations/summary_charts/ folder - Comparison charts\n")
        f.write("6. visualizations/province_distributions/ folder - Individual province distributions\n")
        f.write("\nNote: All ranges are calculated from actual data using IQR method\n")
        f.write("to identify reasonable price ranges excluding outliers.\n")
    
    print(f"✓ Analysis report saved: {report_path}")

# ================================
# MAIN EXECUTION
# ================================
def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("RENTAL HOUSING CANADA - PROVINCE-SPECIFIC ANALYSIS")
    print("="*60)
    print("USING PRE-CLEANED DATA FROM: rental_clean.csv\n")
    
    print("📁 ORGANIZED OUTPUT STRUCTURE:")
    print(f"   Main Directory: {MAIN_OUTPUT_DIR}")
    print(f"   ├── reports/           - Analysis reports")
    print(f"   ├── data/              - CSV data files")
    print(f"   │   └── provinces/     - Individual province data")
    print(f"   └── visualizations/    - PNG chart files")
    print(f"       ├── summary_charts/    - Comparison charts")
    print(f"       └── province_distributions/ - Individual province charts")
    print()
    
    try:
        # Step 1: Load
        df = load_data(INPUT_PATH)
        
        # Step 2: Clean
        df = clean_data(df)
        
        # Step 3: Province Analysis
        province_results = analyze_by_province(df)
        
        # Step 4: Export Province Data
        comparison_df, detailed_df = export_province_data(province_results)
        
        # Step 5: Create Visualizations
        create_province_visualizations(comparison_df, province_results)
        
        # Step 6: Generate Report
        generate_analysis_report(df, province_results, comparison_df)
        
        # Final Summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("="*60)
        print(f"✓ Total listings analyzed: {len(df):,}")
        print(f"✓ Provinces analyzed: {len(province_results)}")
        print(f"✓ Main output directory: {MAIN_OUTPUT_DIR}")
        
        print("\n📊 FILES CREATED:")
        
        print("\n📄 REPORTS:")
        report_files = list(OUTPUT_DIRS['reports'].glob('*'))
        for file in report_files:
            print(f"  • {file.relative_to(MAIN_OUTPUT_DIR)}")
        
        print("\n📈 DATA FILES:")
        # Count province files
        province_csv_files = list(OUTPUT_DIRS['provinces'].glob('*.csv'))
        print(f"  • data/province_comparison.csv")
        print(f"  • data/province_detailed_stats.csv")
        print(f"  • data/provinces/ ({len(province_csv_files)} files)")
        if len(province_csv_files) > 0:
            for file in province_csv_files[:3]:  # Show first 3 as examples
                print(f"    - {file.name}")
            if len(province_csv_files) > 3:
                print(f"    - ... and {len(province_csv_files)-3} more")
        
        print("\n🖼️ VISUALIZATIONS:")
        # Count visualization files
        summary_charts = list(OUTPUT_DIRS['summary_charts'].glob('*.png'))
        province_charts = list(OUTPUT_DIRS['province_charts'].glob('*.png'))
        
        print(f"  • visualizations/summary_charts/ ({len(summary_charts)} charts)")
        for chart in summary_charts:
            print(f"    - {chart.name}")
        
        print(f"  • visualizations/province_distributions/ ({len(province_charts)} charts)")
        if len(province_charts) > 0:
            for chart in province_charts[:3]:  # Show first 3 as examples
                print(f"    - {chart.name}")
            if len(province_charts) > 3:
                print(f"    - ... and {len(province_charts)-3} more")
        
        print("\n✅ NEXT STEPS FOR DASHBOARD:")
        print("  1. Use data/province_comparison.csv for high-level metrics")
        print("  2. Use data/provinces/ folder for detailed province analysis")
        print("  3. Use visualizations/ folder for dashboard graphics")
        print("  4. Reference reports/ for insights and methodology")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

# ================================
# EXECUTION
# ================================
if __name__ == "__main__":
    main()
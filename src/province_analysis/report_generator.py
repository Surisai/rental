"""
Report generation utilities for province analysis.
"""

import pandas as pd
from .config import MAIN_OUTPUT_DIR, OUTPUT_DIRS, INPUT_PATH


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
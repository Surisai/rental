"""
Visualization utilities for province analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from .config import OUTPUT_DIRS


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
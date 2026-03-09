"""
Main orchestration for province analysis.
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .config import INPUT_PATH, MAIN_OUTPUT_DIR, OUTPUT_DIRS
from .data_loader import load_data
from .data_cleaner import clean_data
from .province_analyzer import analyze_by_province
from .data_exporter import export_province_data
from .visualizer import create_province_visualizations
from .report_generator import generate_analysis_report


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
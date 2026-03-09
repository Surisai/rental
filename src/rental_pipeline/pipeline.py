"""
Main pipeline orchestration.
"""

import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .config import RAW_DATA_PATH, CLEAN_DATA_PATH, LOG_PATH
from .logging_utils import initialize_logging, log_step, save_log
from .data_loaders import load_raw_data, save_clean_data
from .data_validators import (
    remove_unnecessary_columns, handle_missing_values, 
    remove_unrealistic_outliers, validate_geographic_data, 
    final_data_quality_check
)
from .text_cleaners import normalize_text_columns, clean_pet_columns
from .numeric_cleaners import clean_price_column, clean_numeric_columns


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
        save_log(log_content, LOG_PATH)
        
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
        save_log(log_content, LOG_PATH)
        
        raise
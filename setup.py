# rental/setup.py
#!/usr/bin/env python3
"""
Setup script for Canadian Rental Analysis Dashboard
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import pandas
        import streamlit
        import plotly
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    requirements_file = Path("requirements.txt")
    
    if requirements_file.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("Requirements installed successfully!")
    else:
        print("Error: requirements.txt not found")

def create_directories():
    """Create necessary directories"""
    print("Creating directory structure...")
    
    directories = [
        "data/raw",
        "data/processed",
        "outputs/province_analysis/reports",
        "outputs/province_analysis/data/provinces",
        "outputs/province_analysis/visualizations/summary_charts",
        "outputs/province_analysis/visualizations/province_distributions",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("Directory structure created!")

def main():
    print("=" * 60)
    print("Canadian Rental Analysis Dashboard Setup")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Check requirements
    print("\nChecking requirements...")
    if not check_requirements():
        print("Some requirements are missing.")
        response = input("Would you like to install them? (y/n): ")
        if response.lower() == 'y':
            install_requirements()
        else:
            print("Please install requirements manually:")
            print("pip install -r requirements.txt")
            return
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Place your rentals.csv file in data/raw/")
    print("2. Run the data cleaning pipeline:")
    print("   python scripts/data_cleaning_pipeline.py")
    print("3. Run province analysis:")
    print("   python scripts/province_analysis.py")
    print("4. Launch the dashboard:")
    print("   streamlit run dashboard/dashboard.py")

if __name__ == "__main__":
    main()
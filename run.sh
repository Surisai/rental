#!/bin/bash

echo "🏠 Canadian Rental Analysis Dashboard - Mac Version"
echo "=================================================="

# Check if data exists
if [ ! -f "data/raw/rentals.csv" ]; then
    echo "❌ ERROR: Please place your rentals.csv file in data/raw/"
    echo "          Then run this script again."
    exit 1
fi

# Run data cleaning if needed
if [ ! -f "data/processed/rentals_clean.csv" ]; then
    echo "📊 Cleaning data..."
    python scripts/data_cleaning_pipeline.py
fi

# Run analysis if needed
if [ ! -f "outputs/province_analysis/data/province_comparison.csv" ]; then
    echo "📈 Running province analysis..."
    python scripts/province_analysis.py
fi

# Launch dashboard
echo "🚀 Launching dashboard..."
echo "👉 Open your browser and go to: http://localhost:8501"
echo "👉 Press Ctrl+C to stop the dashboard"
echo ""
streamlit run dashboard/dashboard.py
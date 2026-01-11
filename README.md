# Canadian Rental Housing Analysis Dashboard

## 📊 Project Overview

This project provides a comprehensive analysis of Canada's rental housing market using 25,000+ listings from RentFaster (June 2024). The analysis includes province-level comparisons, price distributions, pet-friendly policies, and market insights to help understand rental trends across Canada.

**Perfect for:** Data Analyst internship portfolio demonstrating end-to-end data analysis skills.

## 🎯 Key Features

- **Comprehensive Data Analysis**: 25,544 rental listings analyzed across 10 provinces
- **Interactive Dashboard**: Multi-tab interface with filtering capabilities
- **Province Comparison**: Rank provinces by price, listings, pet-friendliness
- **Price Distribution Analysis**: Visualize rental price ranges across provinces
- **Market Insights**: Identify most expensive/affordable provinces, price variability
- **Export Functionality**: Download filtered data and visualizations

## 📈 Key Findings

### Market Overview
- **Overall Average Price**: $2,171/month
- **Most Expensive Province**: British Columbia ($2,524/month)
- **Most Affordable Province**: Newfoundland & Labrador ($1,067/month)
- **Most Active Market**: Alberta (13,752 listings)
- **Most Pet-Friendly**: Saskatchewan (91.5% allow cats)

### Provincial Insights
1. **British Columbia** - Highest average rent, strong pet-friendly policies
2. **Ontario** - Second highest rent, largest variety of listings
3. **Nova Scotia** - Highest price variability (standard deviation: $1,061)
4. **Alberta** - Most listings available, moderate pricing
5. **Saskatchewan** - Most pet-friendly province

## 🏗️ Project Structure
```bash
Rental-Housing-Canada-Analysis/
├── README.md # This file
├── requirements.txt # Python dependencies
├── data/ # Data directory
│ ├── raw/ # Original data
│ │ └── rentals.csv # 25,000+ raw listings
│ └── processed/ # Cleaned data
│ └── rentals_clean.csv # Cleaned dataset
├── scripts/ # Analysis scripts
│ ├── data_cleaning_pipeline.py # Data cleaning pipeline
│ └── province_analysis.py # Province-level analysis
├── outputs/ # Analysis outputs
│ └── province_analysis/
│ ├── reports/ # Text reports
│ │ └── province_analysis_report.txt
│ ├── data/ # CSV data files
│ │ ├── province_comparison.csv
│ │ ├── province_detailed_stats.csv
│ │ └── provinces/ # Individual province data
│ │ ├── alberta_rentals.csv
│ │ ├── british_columbia_rentals.csv
│ │ └── ...
│ └── visualizations/ # PNG chart files
│ ├── summary_charts/ # Comparison charts
│ │ ├── province_analysis_summary.png
│ │ ├── market_size_vs_price.png
│ │ └── outlier_analysis.png
│ └── province_distributions/ # Individual province charts
│ ├── alberta_distribution.png
│ ├── british_columbia_distribution.png
│ └── ...
├── dashboard/ # Streamlit dashboard
│ ├── dashboard.py # Main dashboard app
│ ├── components/ # Dashboard components
│ │ ├── overview.py # Overview tab
│ │ ├── province_comparison.py # Province comparison tab
│ │ ├── province_detail.py # Province detail tab
│ │ ├── market_insights.py # Insights tab
│ │ └── data_explorer.py # Data explorer tab
│ └── utils/ # Utility functions
│ ├── data_loader.py # Data loading utilities
│ ├── visualizations.py # Plotting functions
│ └── filters.py # Filtering utilities
└── logs/ # Log files
└── data_cleaning_log.txt # Cleaning process log
```
## 🚀 Getting Started


### 1. Prerequisites
- Python 3.8+
- pip package manager

### 2. Installation
```bash
# Clone repository
git clone <repository-url>
cd Rental-Housing-Canada-Analysis

# Install dependencies
pip install -r requirements.txt

### 3.  Run Data Pipeline
```bash
# Step 1: Clean the raw data
python3 scripts/data_cleaning_pipeline.py

# Step 2: Run province analysis
python3 scripts/province_analysis.py

### 4. Launch Dashboard
```bash
# Run the Streamlit dashboard
streamlit run dashboard/dashboard.py
``

# Sample Data

For demonstration purposes only.
Full dataset (25,544 rows) available upon request.
EOF
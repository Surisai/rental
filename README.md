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
python3 scripts/data_cleanning_pipeline.py

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

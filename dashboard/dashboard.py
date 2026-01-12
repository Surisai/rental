from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import subprocess
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

#Configuration add centralize all paths 
BASE_DIR = Path(__file__).resolve().parents[1]

PROVINCE_COMPARISON_PATH = (
    BASE_DIR/"outputs/province_analysis/data/province_comparison.csv"
)


ANALYSIS_SCRIPT = BASE_DIR /"scripts/province_analysis.py"
# Page configuration
st.set_page_config(
    page_title="Canadian Rental Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
#status handler 
def ensure_analysis_data():
    """
    Ensures required analysis data exists.
    Returns True if data is ready, False otherwise.
    """

    if PROVINCE_COMPARISON_PATH.exists():
        return True

    st.warning("📊 Analysis data is not available yet.")

    st.markdown("""
    **What’s happening?**
    - This app requires pre-computed analysis results.
    - Streamlit Cloud starts with a clean environment.
    
    **What can you do?**
    - Click the button below to generate the data.
    """)

    if st.button("▶ Run Analysis Pipeline"):
        with st.spinner("Running analysis... this may take a moment"):
            try:
                result = subprocess.run(
                    ["python", str(ANALYSIS_SCRIPT)],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    st.error("❌ Analysis script failed to run.")
                    st.markdown("**Error details:**")
                    st.code(result.stderr)
                    st.stop()

                st.success("✅ Analysis completed successfully!")
                st.rerun()

            except Exception as e:
                st.error("❌ Analysis failed unexpectedly.")
                st.exception(e)

    return False

#Checking is the data is missing
if not ensure_analysis_data():
    st.stop()

#Safe to load data 
df = pd.read_csv(PROVINCE_COMPARISON_PATH)

#Rest of the dashboard code 
st.title("Rental Market Analysis Dashboard")
st.dataframe(df)

# Custom CSS
def add_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
def create_sidebar():
    with st.sidebar:
        st.markdown("## 🏠 Canadian Rentals")
        st.markdown("---")
        
        # Dataset info
        st.markdown("### 📊 Dataset")
        st.info("""
        **25,544 listings**  
        **10 provinces**  
        **June 2024 data**  
        Source: RentFaster
        """)
        
        st.markdown("---")
        
        # Quick navigation
        st.markdown("### 📑 Quick Links")
        if st.button("📈 Go to Overview"):
            st.session_state.active_tab = "Overview"
        if st.button("🏙️ Compare Provinces"):
            st.session_state.active_tab = "Province Comparison"
        if st.button("🔍 Explore Data"):
            st.session_state.active_tab = "Data Explorer"
        
        st.markdown("---")
        
        # Data info
        st.markdown("### ℹ️ Info")
        st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y')}")

# Load data function
@st.cache_data
def load_data():
    """Load analysis data"""
    try:
        # Load comparison data
        comparison_path = Path("outputs/province_analysis/data/province_comparison.csv")
        comparison_df = pd.read_csv(comparison_path)
        
        # Load detailed stats
        detailed_path = Path("outputs/province_analysis/data/province_detailed_stats.csv")
        detailed_df = pd.read_csv(detailed_path)
        
        # Load province data
        province_data_dict = {}
        provinces_path = Path("outputs/province_analysis/data/provinces")
        
        for province_file in provinces_path.glob("*_rentals.csv"):
            province_name = province_file.stem.replace("_rentals", "").replace("_", " ").title()
            province_df = pd.read_csv(province_file)
            province_data_dict[province_name] = province_df
        
        return comparison_df, detailed_df, province_data_dict
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

# Visualization functions
def create_price_chart(comparison_df):
    """Create average price chart"""
    fig = px.bar(comparison_df.sort_values('mean_price'), 
                 x='mean_price', y='province',
                 orientation='h',
                 title='Average Monthly Rent by Province',
                 color='mean_price',
                 color_continuous_scale='viridis')
    
    fig.update_layout(
        xaxis_title="Average Rent (CAD)",
        yaxis_title="Province",
        coloraxis_showscale=False
    )
    return fig

def create_listings_chart(comparison_df):
    """Create listings count chart"""
    fig = px.bar(comparison_df.sort_values('total_listings'), 
                 x='total_listings', y='province',
                 orientation='h',
                 title='Number of Listings by Province',
                 color='total_listings',
                 color_continuous_scale='blues')
    
    fig.update_layout(
        xaxis_title="Number of Listings",
        yaxis_title="Province",
        coloraxis_showscale=False
    )
    return fig

def create_pet_chart(comparison_df):
    """Create pet-friendly chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comparison_df['province'],
        y=comparison_df['cats_allowed_pct'],
        name='Cats Allowed',
        marker_color='orange'
    ))
    
    fig.add_trace(go.Bar(
        x=comparison_df['province'],
        y=comparison_df['dogs_allowed_pct'],
        name='Dogs Allowed',
        marker_color='blue'
    ))
    
    fig.update_layout(
        title='Pet-Friendly Listings by Province',
        xaxis_title='Province',
        yaxis_title='Percentage (%)',
        barmode='group',
        xaxis_tickangle=-45
    )
    return fig

# Overview tab
def create_overview_tab(comparison_df):
    """Create overview tab content"""
    st.markdown('<h2 class="main-header">📈 Market Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = comparison_df['mean_price'].mean()
        st.metric("National Average", f"${avg_price:,.0f}")
    
    with col2:
        total_listings = comparison_df['total_listings'].sum()
        st.metric("Total Listings", f"{total_listings:,}")
    
    with col3:
        most_expensive = comparison_df.loc[comparison_df['mean_price'].idxmax()]
        st.metric("Most Expensive", most_expensive['province'])
    
    with col4:
        most_affordable = comparison_df.loc[comparison_df['mean_price'].idxmin()]
        st.metric("Most Affordable", most_affordable['province'])
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_price_chart(comparison_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_listings_chart(comparison_df), use_container_width=True)
    
    # Insights
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        with st.container(border=True):
            st.markdown("### 🏆 Top 3 Provinces by Price")
            top3 = comparison_df.nlargest(3, 'mean_price')
            for i, (_, row) in enumerate(top3.iterrows(), 1):
                st.markdown(f"""
                **{i}. {row['province']}**
                - Rent: **${row['mean_price']:,.0f}**
                - Listings: {row['total_listings']:,}
                """)
    
    with insight_col2:
        with st.container(border=True):
            st.markdown("### 🏆 Top 3 by Listings")
            top3_listings = comparison_df.nlargest(3, 'total_listings')
            for i, (_, row) in enumerate(top3_listings.iterrows(), 1):
                st.markdown(f"""
                **{i}. {row['province']}**
                - Listings: **{row['total_listings']:,}**
                - Avg Rent: ${row['mean_price']:,.0f}
                """)

# Province comparison tab
def create_comparison_tab(comparison_df):
    """Create province comparison tab"""
    st.markdown('<h2 class="main-header">🏙️ Province Comparison</h2>', unsafe_allow_html=True)
    
    # Selection
    col1, col2 = st.columns(2)
    
    with col1:
        provinces = comparison_df['province'].tolist()
        selected_provinces = st.multiselect(
            "Select provinces to compare:",
            provinces,
            default=provinces[:3]
        )
    
    with col2:
        metric = st.selectbox(
            "Select metric to compare:",
            ['mean_price', 'total_listings', 'cats_allowed_pct', 'dogs_allowed_pct'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    if selected_provinces:
        filtered_df = comparison_df[comparison_df['province'].isin(selected_provinces)]
        
        # Comparison chart
        fig = px.bar(filtered_df, x='province', y=metric,
                    title=f'{metric.replace("_", " ").title()} Comparison',
                    color='province')
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        st.subheader("Comparison Table")
        display_cols = ['province', 'mean_price', 'total_listings', 
                       'cats_allowed_pct', 'dogs_allowed_pct']
        display_df = filtered_df[display_cols].copy()
        display_df.columns = [col.replace('_', ' ').title() for col in display_cols]
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Please select at least one province to compare.")

# Province detail tab
def create_detail_tab(comparison_df, province_data_dict):
    """Create province detail tab"""
    st.markdown('<h2 class="main-header">🔍 Province Detail Analysis</h2>', unsafe_allow_html=True)
    
    # Province selector
    selected_province = st.selectbox(
        "Select a province for detailed analysis:",
        comparison_df['province'].tolist()
    )
    
    if selected_province:
        province_data = comparison_df[comparison_df['province'] == selected_province].iloc[0]
        
        # Province metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Rent", f"${province_data['mean_price']:,.0f}")
        
        with col2:
            st.metric("Listings", f"{province_data['total_listings']:,}")
        
        with col3:
            st.metric("Cities", f"{province_data.get('city_count', 'N/A')}")
        
        with col4:
            st.metric("Outliers", f"{province_data['outlier_percentage']:.1f}%")
        
        st.markdown("---")
        
        # Detailed info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Price Information")
            st.markdown(f"""
            - **Minimum:** ${province_data['min_price']:,.0f}
            - **Maximum:** ${province_data['max_price']:,.0f}
            - **Median:** ${province_data['median_price']:,.0f}
            - **Standard Deviation:** ${province_data['price_std']:,.0f}
            - **Reasonable Range:** ${province_data.get('reasonable_min', 0):,.0f} - ${province_data.get('reasonable_max', 0):,.0f}
            """)
        
        with col2:
            if 'cats_allowed_pct' in province_data:
                st.subheader("🐾 Pet Policy")
                st.markdown(f"""
                - **Cats Allowed:** {province_data['cats_allowed_pct']:.1f}%
                - **Dogs Allowed:** {province_data['dogs_allowed_pct']:.1f}%
                """)
            
            if 'avg_beds' in province_data:
                st.subheader("🏠 Property Info")
                st.markdown(f"""
                - **Avg Bedrooms:** {province_data['avg_beds']:.1f}
                - **Avg Size:** {province_data.get('avg_size', 'N/A')} sq ft
                """)
        
        # Show sample data if available
        if selected_province in province_data_dict:
            st.markdown("---")
            st.subheader("📋 Sample Listings")
            sample_data = province_data_dict[selected_province].head(10)
            st.dataframe(sample_data[['city', 'price', 'beds', 'type']], use_container_width=True)

# Data explorer tab
def create_explorer_tab(province_data_dict):
    """Create data explorer tab"""
    st.markdown('<h2 class="main-header">📊 Data Explorer</h2>', unsafe_allow_html=True)
    
    # Province selector
    available_provinces = list(province_data_dict.keys())
    selected_province = st.selectbox(
        "Select province to explore:",
        available_provinces
    )
    
    if selected_province and selected_province in province_data_dict:
        df = province_data_dict[selected_province]
        
        # Filters
        st.subheader("🔍 Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price_range = st.slider(
                "Price Range (CAD)",
                min_value=int(df['price'].min()),
                max_value=int(df['price'].max()),
                value=(int(df['price'].min()), int(df['price'].max()))
            )
        
        with col2:
            if 'beds' in df.columns:
                bed_options = sorted(df['beds'].dropna().unique())
                selected_beds = st.multiselect(
                    "Bedrooms",
                    bed_options,
                    default=bed_options
                )
            else:
                selected_beds = None
        
        with col3:
            if 'type' in df.columns:
                type_options = df['type'].unique()
                selected_types = st.multiselect(
                    "Property Type",
                    type_options,
                    default=type_options[:3] if len(type_options) > 3 else type_options
                )
            else:
                selected_types = None
        
        # Apply filters
        filtered_df = df.copy()
        filtered_df = filtered_df[
            (filtered_df['price'] >= price_range[0]) & 
            (filtered_df['price'] <= price_range[1])
        ]
        
        if selected_beds:
            filtered_df = filtered_df[filtered_df['beds'].isin(selected_beds)]
        
        if selected_types:
            filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
        
        # Display data
        st.subheader(f"📋 {selected_province} Listings ({len(filtered_df):,} found)")
        
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Price", f"${filtered_df['price'].mean():,.0f}")
        with col2:
            st.metric("Min Price", f"${filtered_df['price'].min():,.0f}")
        with col3:
            st.metric("Max Price", f"${filtered_df['price'].max():,.0f}")
        
        # Show data table
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export option
        if st.button("📥 Download Filtered Data (CSV)"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_province.lower().replace(' ', '_')}_filtered.csv",
                mime="text/csv"
            )


# Main app
def main():
    # Add custom CSS
    add_custom_css()
    
    # Create sidebar
    create_sidebar()
    
    # Load data
    with st.spinner("Loading data..."):
        comparison_df, detailed_df, province_data_dict = load_data()
    
    if comparison_df.empty:
        st.error("No data found. Please run the analysis scripts first.")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🏙️ Compare",
        "🔍 Details",
        "📊 Explore"
    ])
    
    with tab1:
        create_overview_tab(comparison_df)
    
    with tab2:
        create_comparison_tab(comparison_df)
    
    with tab3:
        create_detail_tab(comparison_df, province_data_dict)
    
    with tab4:
        create_explorer_tab(province_data_dict)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Data:** RentFaster Canada")
    with col2:
        st.markdown("**Listings:** 25,544")
    with col3:
        st.markdown(f"**Updated:** {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Medicaid Drug Spend Risk Forecasting - Streamlit Cloud Demo Dashboard
Lightweight demo version that works with sample data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Medicaid Drug Spend Risk Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Medicaid Drug Spend Risk Forecasting")
st.markdown("Interactive dashboard for analyzing Medicaid drug spend trends and risk assessment")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def generate_sample_data():
    """Generate realistic sample data for demonstration."""
    
    # Sample states and drugs
    states = ['CA', 'TX', 'NY', 'FL', 'PA', 'OH', 'IL', 'GA', 'NC', 'MI']
    drug_names = [
        'Lisinopril', 'Metformin', 'Amlodipine', 'Omeprazole', 'Losartan',
        'Simvastatin', 'Atorvastatin', 'Metoprolol', 'Hydrochlorothiazide', 'Furosemide',
        'Gabapentin', 'Tramadol', 'Sertraline', 'Bupropion', 'Escitalopram',
        'Albuterol', 'Fluticasone', 'Montelukast', 'Cetirizine', 'Loratadine'
    ]
    
    # Generate quarterly data for 2020-2024
    quarters = []
    for year in range(2020, 2025):
        for quarter in range(1, 5):
            quarters.append((year, quarter))
    
    # Generate sample data
    data = []
    for state in states:
        for drug in drug_names:
            for year, quarter in quarters:
                # Base values with realistic variations
                base_spend = np.random.uniform(10000, 500000)
                base_units = np.random.uniform(1000, 50000)
                base_scripts = np.random.uniform(100, 5000)
                
                # Add trend and seasonality
                trend_factor = 1 + (year - 2020) * 0.1 + (quarter - 1) * 0.02
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (quarter - 1) / 4)
                
                # Add random noise
                noise = np.random.uniform(0.8, 1.2)
                
                total_spend = base_spend * trend_factor * seasonal_factor * noise
                total_units = base_units * trend_factor * seasonal_factor * noise
                total_scripts = base_scripts * trend_factor * seasonal_factor * noise
                
                # Calculate derived metrics
                spend_per_script = total_spend / total_scripts if total_scripts > 0 else 0
                spend_per_unit = total_spend / total_units if total_units > 0 else 0
                
                data.append({
                    'state': state,
                    'ndc11_normalized': f"{np.random.randint(10000000000, 99999999999)}",
                    'product_name_fda10': drug,
                    'year': year,
                    'quarter': quarter,
                    'quarter_start': date(year, (quarter - 1) * 3 + 1, 1),
                    'total_spend': round(total_spend, 2),
                    'total_units': round(total_units, 2),
                    'total_scripts': round(total_scripts),
                    'spend_per_script': round(spend_per_script, 2),
                    'spend_per_unit': round(spend_per_unit, 2)
                })
    
    return pd.DataFrame(data)


def calculate_trend_features(df):
    """Calculate trend features for the sample data."""
    
    # Sort by state, drug, and quarter
    df = df.sort_values(['state', 'product_name_fda10', 'quarter_start']).reset_index(drop=True)
    
    # Calculate quarter-over-quarter changes
    df['qoq_spend_change'] = df.groupby(['state', 'product_name_fda10'])['total_spend'].pct_change()
    df['qoq_units_change'] = df.groupby(['state', 'product_name_fda10'])['total_units'].pct_change()
    df['qoq_scripts_change'] = df.groupby(['state', 'product_name_fda10'])['total_scripts'].pct_change()
    
    # Calculate rolling statistics (4-quarter window) - fix indexing issue
    rolling_avg = df.groupby(['state', 'product_name_fda10'])['total_spend'].rolling(4, min_periods=1).mean()
    rolling_std = df.groupby(['state', 'product_name_fda10'])['total_spend'].rolling(4, min_periods=1).std()
    
    # Reset index to align with original DataFrame
    df['rolling_4q_spend_avg'] = rolling_avg.reset_index(level=[0,1], drop=True)
    df['rolling_4q_spend_std'] = rolling_std.reset_index(level=[0,1], drop=True)
    
    # Coefficient of variation
    df['cv_spend'] = df['rolling_4q_spend_std'] / df['rolling_4q_spend_avg']
    
    # Rolling growth rates - fix indexing issue
    def calculate_growth(group):
        if len(group) >= 4:
            return (group.iloc[-1] - group.iloc[0]) / group.iloc[0] if group.iloc[0] != 0 else 0
        else:
            return 0
    
    rolling_growth = df.groupby(['state', 'product_name_fda10'])['total_spend'].rolling(4, min_periods=1).apply(
        calculate_growth, raw=True
    )
    df['rolling_4q_growth_spend'] = rolling_growth.reset_index(level=[0,1], drop=True)
    
    return df


def calculate_risk_scores(df):
    """Calculate risk scores based on the latest quarter data."""
    
    # Get latest quarter data
    latest_quarter = df['quarter_start'].max()
    latest_data = df[df['quarter_start'] == latest_quarter].copy()
    
    # Calculate risk flags
    latest_data['high_cost'] = latest_data.groupby('state')['total_spend'].transform(
        lambda x: x >= x.quantile(0.75)
    )
    
    latest_data['high_vol'] = (latest_data['cv_spend'] > 1.0) | (abs(latest_data['qoq_spend_change']) > 0.35)
    latest_data['fast_riser'] = latest_data['rolling_4q_growth_spend'] > 0.25
    
    # Calculate risk level
    latest_data['risk_level'] = 'low'
    latest_data.loc[
        latest_data['high_cost'] & (latest_data['high_vol'] | latest_data['fast_riser']), 'risk_level'
    ] = 'high'
    latest_data.loc[
        (latest_data['high_cost'] | latest_data['high_vol'] | latest_data['fast_riser']) & 
        (latest_data['risk_level'] != 'high'), 'risk_level'
    ] = 'medium'
    
    # Calculate risk score (0-100)
    latest_data['risk_score'] = (
        latest_data['high_cost'].astype(int) * 30 +
        latest_data['high_vol'].astype(int) * 25 +
        latest_data['fast_riser'].astype(int) * 25 +
        (latest_data['cv_spend'] * 20).clip(0, 20)
    ).round(1)
    
    return latest_data


def load_demo_data():
    """Load and process demo data."""
    if not st.session_state.data_loaded:
        with st.spinner("Generating sample data..."):
            # Generate sample data
            df = generate_sample_data()
            
            # Calculate features
            df = calculate_trend_features(df)
            
            # Calculate risks
            risk_df = calculate_risk_scores(df)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.risk_df = risk_df
            st.session_state.data_loaded = True
            
            st.success("Sample data loaded successfully!")
    
    return st.session_state.df, st.session_state.risk_df


# Sidebar controls
st.sidebar.header("Controls")

# Data loading section
st.sidebar.subheader("Data Management")
if st.sidebar.button("Load Demo Data", type="primary"):
    df, risk_df = load_demo_data()

# Forecasting section
st.sidebar.subheader("Forecasting")
metric = st.sidebar.selectbox("Forecast Metric", ["spend", "units"])
top_k = st.sidebar.slider("Top K per State", 10, 100, 50, 10)

# Filtering section
st.sidebar.subheader("Filters")
if st.session_state.data_loaded:
    available_states = sorted(df['state'].unique())
    selected_states = st.sidebar.multiselect(
        "Select States",
        available_states,
        default=available_states[:5]
    )
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(date(2020, 1, 1), date(2024, 12, 31)),
        min_value=date(2020, 1, 1),
        max_value=date(2024, 12, 31)
    )
else:
    selected_states = []
    date_range = (date(2020, 1, 1), date(2024, 12, 31))


# Main content
if not st.session_state.data_loaded:
    st.info("Click 'Load Demo Data' to begin exploring the dashboard")
    
    # Show app description
    st.markdown("""
    ## About This Dashboard
    
    This is a **demonstration version** of the Medicaid Drug Spend Risk Forecasting dashboard.
    
    ### What You'll See:
    - **Sample Data**: Realistic Medicaid drug utilization data for 10 states and 20 common drugs
    - **Risk Assessment**: Automated risk scoring based on cost, volatility, and growth trends
    - **Interactive Visualizations**: Charts and tables for exploring drug spend patterns
    - **Forecasting Interface**: Tools for analyzing future trends (demo mode)
    
    ### How to Use:
    1. Click **"Load Demo Data"** to generate sample data
    2. Use the sidebar filters to explore specific states or time periods
    3. Navigate through different dashboard sections
    4. Interact with charts and tables
    
    ### Note:
    This demo uses simulated data to showcase the dashboard functionality.
    For production use with real data, the full pipeline would be run locally.
    """)

else:
    # Load data
    df, risk_df = load_demo_data()
    
    # Apply filters
    if selected_states:
        df_filtered = df[df['state'].isin(selected_states)].copy()
        risk_df_filtered = risk_df[risk_df['state'].isin(selected_states)].copy()
    else:
        df_filtered = df.copy()
        risk_df_filtered = risk_df.copy()
    
    # Overview section
    st.header("Overview")
    
    # KPI tiles
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_spend = df_filtered['total_spend'].sum()
        st.metric("Total Spend", f"${total_spend:,.0f}")
    
    with col2:
        unique_drugs = df_filtered['ndc11_normalized'].nunique()
        st.metric("Unique Drugs", f"{unique_drugs:,}")
    
    with col3:
        unique_states = df_filtered['state'].nunique()
        st.metric("States", f"{unique_states}")
    
    with col4:
        latest_quarter = df_filtered['quarter_start'].max()
        st.metric("Latest Quarter", str(latest_quarter)[:7])
    
    # Top spend by drug
    st.subheader("Top 10 Drugs by Spend (Latest Quarter)")
    latest_q = df_filtered['quarter_start'].max()
    top_spend = df_filtered[df_filtered['quarter_start'] == latest_q].nlargest(10, 'total_spend')
    
    fig = px.bar(
        top_spend,
        x="product_name_fda10",
        y="total_spend",
        color="state",
        title="Top 10 Drugs by Spend",
        labels={"total_spend": "Total Spend ($)", "product_name_fda10": "Drug Name"}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top risers
    st.subheader("Top Risers by Quarter-over-Quarter Change")
    latest_risers = df_filtered[df_filtered['quarter_start'] == latest_q].nlargest(10, 'qoq_spend_change')
    
    fig = px.bar(
        latest_risers,
        x="product_name_fda10",
        y="qoq_spend_change",
        color="state",
        title="Top 10 Drugs by QoQ Spend Change",
        labels={"qoq_spend_change": "QoQ Change (%)", "product_name_fda10": "Drug Name"}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk watchlist
    st.header("Risk Watchlist")
    
    # Risk summary by level
    risk_summary = risk_df_filtered['risk_level'].value_counts().reset_index()
    risk_summary.columns = ['risk_level', 'drug_count']
    
    fig = px.pie(
        risk_summary,
        values="drug_count",
        names="risk_level",
        title="Risk Distribution by Level"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # High risk table
    st.subheader("High and Medium Risk Drugs")
    high_risk = risk_df_filtered[risk_df_filtered['risk_level'].isin(['high', 'medium'])].nlargest(20, 'risk_score')
    
    st.dataframe(
        high_risk[["state", "product_name_fda10", "total_spend", "risk_level", "risk_score", "qoq_spend_change"]],
        use_container_width=True
    )
    
    # Download button
    csv = high_risk.to_csv(index=False)
    st.download_button(
        label="Download Risk Watchlist CSV",
        data=csv,
        file_name=f"risk_watchlist_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Trends by Drug section
    st.header("Trends by Drug")
    
    # Drug selection
    col1, col2 = st.columns(2)
    
    with col1:
        if selected_states:
            selected_state = st.selectbox("Select State", selected_states)
        else:
            selected_state = st.selectbox("Select State", sorted(df['state'].unique()))
    
    with col2:
        # Get drugs for selected state
        state_drugs = df_filtered[df_filtered['state'] == selected_state]['product_name_fda10'].unique()
        selected_drug = st.selectbox("Select Drug", sorted(state_drugs))
    
    if selected_drug:
        # Get historical data
        drug_history = df_filtered[
            (df_filtered['state'] == selected_state) & 
            (df_filtered['product_name_fda10'] == selected_drug)
        ].sort_values('quarter_start')
        
        if not drug_history.empty:
            # Historical trends
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Historical Spend", "Historical Units"),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=drug_history["quarter_start"], y=drug_history["total_spend"], 
                          mode="lines+markers", name="Spend"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=drug_history["quarter_start"], y=drug_history["total_units"], 
                          mode="lines+markers", name="Units"),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text=f"Historical Trends for {selected_drug}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show trend statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest QoQ Change", f"{drug_history['qoq_spend_change'].iloc[-1]:.1%}")
            with col2:
                st.metric("4Q Growth Rate", f"{drug_history['rolling_4q_growth_spend'].iloc[-1]:.1%}")
            with col3:
                st.metric("Volatility (CV)", f"{drug_history['cv_spend'].iloc[-1]:.2f}")
    
    # State comparison
    st.header("State Comparison")
    
    # Select metric for comparison
    comparison_metric = st.selectbox("Compare by", ["total_spend", "total_units", "total_scripts"])
    
    # Get latest quarter data for comparison
    latest_comparison = df_filtered[df_filtered['quarter_start'] == latest_q]
    
    if not latest_comparison.empty:
        # State comparison chart
        state_avg = latest_comparison.groupby('state')[comparison_metric].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=state_avg.index,
            y=state_avg.values,
            title=f"Average {comparison_metric.replace('_', ' ').title()} by State",
            labels={"x": "State", "y": comparison_metric.replace('_', ' ').title()}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top states table
        st.subheader(f"Top 10 States by {comparison_metric.replace('_', ' ').title()}")
        top_states = state_avg.head(10).reset_index()
        top_states.columns = ['State', comparison_metric.replace('_', ' ').title()]
        st.dataframe(top_states, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("Medicaid Drug Spend Risk Forecasting Dashboard | Demo Version | Built with Streamlit and Plotly")

# Add info about demo data
if st.session_state.data_loaded:
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Demo Data Info:**
    - 10 states (CA, TX, NY, FL, PA, OH, IL, GA, NC, MI)
    - 20 common drugs
    - 2020-2024 quarterly data
    - Realistic trends and seasonality
    """)

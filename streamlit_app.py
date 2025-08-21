#!/usr/bin/env python3
"""
Medicaid Drug Spend Risk Forecasting - Streamlit Cloud Deployment
Main entry point for Streamlit Cloud deployment.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.io import get_spark, read_raw_csvs, write_parquet, read_parquet
from src.schema import detect_columns, apply_schema, validate_ranges
from src.transforms import quarterly_aggregates, cleanse_data, add_time_dimensions
from src.features import trend_features, add_lag_features
from src.risk import flag_risks, get_latest_quarter_risks
from src.forecast import select_forecast_universe, run_batch_forecasts, save_forecasts, load_forecasts
from src.sql import create_temp_views, get_top_spend_by_state_latest, get_top_risers_by_state_latest, get_history_by_state_ndc, get_high_risk_drugs, get_risk_summary_by_state
from pyspark.sql.functions import col


# Page configuration
st.set_page_config(
    page_title="Medicaid Drug Spend Risk Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Medicaid Drug Spend Risk Forecasting")
st.markdown("Process Medicaid claims data, assess risks, and forecast drug spend trends")

# Initialize session state
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'spark_initialized' not in st.session_state:
    st.session_state.spark_initialized = False


@st.cache_resource
def initialize_spark():
    """Initialize Spark session with caching."""
    return get_spark("medicaid-risk-streamlit")


def process_data_pipeline():
    """Process the complete data pipeline."""
    try:
        with st.spinner("Initializing Spark session..."):
            spark = initialize_spark()
            st.session_state.spark_initialized = True
        
        with st.spinner("Reading raw CSV files..."):
            raw_df = read_raw_csvs(spark, "data/raw")
            st.write(f"Loaded {raw_df.count():,} records from raw data")
        
        with st.spinner("Detecting and applying schema..."):
            column_mapping = detect_columns(raw_df)
            st.write(f"Detected {len(column_mapping)} columns")
            
            schema_df = apply_schema(raw_df, column_mapping)
            validated_df = validate_ranges(schema_df)
            st.write(f"Schema applied and validation completed")
        
        with st.spinner("Cleansing data..."):
            cleansed_df = cleanse_data(validated_df)
            st.write(f"Data cleansing completed, {cleansed_df.count():,} records remaining")
        
        with st.spinner("Computing quarterly aggregates..."):
            aggregated_df = quarterly_aggregates(cleansed_df)
            aggregated_df = add_time_dimensions(aggregated_df)
            
            # Save aggregates
            write_parquet(aggregated_df, "data/processed/aggregates.parquet")
            st.write("Quarterly aggregates computed and saved")
        
        with st.spinner("Calculating trend features..."):
            features_df = trend_features(aggregated_df)
            features_df = add_lag_features(features_df)
            
            # Save features
            write_parquet(features_df, "data/processed/features.parquet")
            st.write("Trend features calculated and saved")
        
        with st.spinner("Assessing risks..."):
            risk_df = flag_risks(features_df, by_state=True)
            
            # Save risks
            write_parquet(risk_df, "data/processed/risks.parquet")
            st.write("Risk assessment completed and saved")
        
        with st.spinner("Creating SQL views..."):
            create_temp_views(spark, aggregated_df, features_df, risk_df)
            st.write("SQL views created successfully")
        
        st.session_state.data_processed = True
        st.success("Data processing pipeline completed successfully!")
        
        return spark, aggregated_df, features_df, risk_df
        
    except Exception as e:
        st.error(f"Error in data processing pipeline: {str(e)}")
        return None, None, None, None


def run_forecasts(spark, features_df, metric="spend", top_k=50):
    """Run forecasting pipeline."""
    try:
        with st.spinner("Selecting forecast universe..."):
            universe = select_forecast_universe(features_df, top_k, metric)
            st.write(f"Selected {universe.count():,} drugs for forecasting")
        
        with st.spinner("Running Prophet forecasts..."):
            forecasts_df = run_batch_forecasts(features_df, universe, metric)
            
            # Save forecasts
            save_forecasts(forecasts_df, f"data/processed/forecasts_{metric}.parquet")
            st.write(f"Forecasts completed for {len(forecasts_df)} drug-quarter combinations")
        
        return forecasts_df
        
    except Exception as e:
        st.error(f"Error in forecasting pipeline: {str(e)}")
        return None


# Sidebar controls
st.sidebar.header("Controls")

# Data processing section
st.sidebar.subheader("Data Processing")
if st.sidebar.button("Process Data Pipeline", type="primary"):
    spark, agg_df, feat_df, risk_df = process_data_pipeline()

# Forecasting section
st.sidebar.subheader("Forecasting")
metric = st.sidebar.selectbox("Forecast Metric", ["spend", "units"])
top_k = st.sidebar.slider("Top K per State", 10, 100, 50, 10)

if st.sidebar.button("Run Forecasts"):
    if st.session_state.spark_initialized:
        spark = initialize_spark()
        features_df = read_parquet("data/processed/features.parquet")
        forecasts_df = run_forecasts(spark, features_df, metric, top_k)
        if forecasts_df is not None:
            st.session_state.forecasts = forecasts_df

# Filtering section
st.sidebar.subheader("Filters")
available_states = []
if os.path.exists("data/processed/aggregates.parquet"):
    try:
        spark = initialize_spark()
        agg_df = read_parquet("data/processed/aggregates.parquet")
        available_states = agg_df.select("state").distinct().orderBy("state").toPandas()["state"].tolist()
    except:
        available_states = []

selected_states = st.sidebar.multiselect(
    "Select States",
    available_states,
    default=available_states[:5] if len(available_states) > 0 else []
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(date(2020, 1, 1), date(2024, 12, 31)),
    min_value=date(2010, 1, 1),
    max_value=date(2030, 12, 31)
)


# Main content
if not st.session_state.data_processed:
    st.info("Click 'Process Data Pipeline' to begin data processing")
    
    # Show sample data if available
    if os.path.exists("data/raw"):
        raw_files = [f for f in os.listdir("data/raw") if f.endswith('.csv')]
        if raw_files:
            st.write(f"Found {len(raw_files)} CSV files in data/raw/")
            st.write("Files:", raw_files)
        else:
            st.warning("No CSV files found in data/raw/ directory")
    else:
        st.warning("data/raw/ directory not found")

else:
    # Load processed data
    try:
        spark = initialize_spark()
        agg_df = read_parquet("data/processed/aggregates.parquet")
        features_df = read_parquet("data/processed/features.parquet")
        risk_df = read_parquet("data/processed/risks.parquet")
        
        # Apply filters
        if selected_states:
            agg_df = agg_df.filter(col("state").isin(selected_states))
            features_df = features_df.filter(col("state").isin(selected_states))
            risk_df = risk_df.filter(col("state").isin(selected_states))
        
        # Overview section
        st.header("Overview")
        
        # KPI tiles
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_spend = agg_df.agg({"total_spend": "sum"}).collect()[0][0]
            st.metric("Total Spend", f"${total_spend:,.0f}")
        
        with col2:
            unique_drugs = agg_df.select("ndc11_normalized").distinct().count()
            st.metric("Unique Drugs", f"{unique_drugs:,}")
        
        with col3:
            unique_states = agg_df.select("state").distinct().count()
            st.metric("States", f"{unique_states}")
        
        with col4:
            latest_quarter = agg_df.agg({"quarter_start": "max"}).collect()[0][0]
            st.metric("Latest Quarter", str(latest_quarter)[:7])
        
        # Top spend by drug
        st.subheader("Top 10 Drugs by Spend (Latest Quarter)")
        top_spend = get_top_spend_by_state_latest(spark, limit=10)
        if not top_spend.isEmpty():
            top_spend_pd = top_spend.toPandas()
            
            fig = px.bar(
                top_spend_pd.head(10),
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
        top_risers = get_top_risers_by_state_latest(spark, limit=10)
        if not top_risers.isEmpty():
            top_risers_pd = top_risers.toPandas()
            
            fig = px.bar(
                top_risers_pd.head(10),
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
        high_risk = get_high_risk_drugs(spark, limit=20)
        if not high_risk.isEmpty():
            high_risk_pd = high_risk.toPandas()
            
            # Risk summary by level
            risk_summary = get_risk_summary_by_state(spark)
            if not risk_summary.isEmpty():
                risk_summary_pd = risk_summary.toPandas()
                
                fig = px.pie(
                    risk_summary_pd,
                    values="drug_count",
                    names="risk_level",
                    title="Risk Distribution by Level"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # High risk table
            st.subheader("High and Medium Risk Drugs")
            st.dataframe(
                high_risk_pd[["state", "product_name_fda10", "total_spend", "risk_level", "risk_score", "qoq_spend_change"]],
                use_container_width=True
            )
            
            # Download button
            csv = high_risk_pd.to_csv(index=False)
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
                selected_state = st.selectbox("Select State", available_states)
        
        with col2:
            # Get drugs for selected state
            state_drugs = agg_df.filter(col("state") == selected_state).select("ndc11_normalized", "product_name_fda10").distinct()
            if not state_drugs.isEmpty():
                state_drugs_pd = state_drugs.toPandas()
                selected_drug = st.selectbox("Select Drug", state_drugs_pd["ndc11_normalized"].tolist())
            else:
                selected_drug = None
        
        if selected_drug:
            # Get historical data
            history = get_history_by_state_ndc(spark, selected_state, selected_drug)
            if not history.isEmpty():
                history_pd = history.toPandas()
                
                # Historical trends
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Historical Spend", "Historical Units"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(x=history_pd["quarter_start"], y=history_pd["total_spend"], 
                              mode="lines+markers", name="Spend"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=history_pd["quarter_start"], y=history_pd["total_spend"], 
                              mode="lines+markers", name="Units"),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, title_text=f"Historical Trends for {selected_drug}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast if available
                forecast_file = f"data/processed/forecasts_{metric}.parquet"
                if os.path.exists(forecast_file):
                    forecasts = load_forecasts(forecast_file)
                    drug_forecasts = forecasts[
                        (forecasts["state"] == selected_state) & 
                        (forecasts["ndc11"] == selected_drug)
                    ]
                    
                    if not drug_forecasts.empty:
                        st.subheader("Forecast (Next 4 Quarters)")
                        
                        # Combine historical and forecast data
                        forecast_plot = pd.DataFrame({
                            'quarter_start': pd.concat([
                                history_pd['quarter_start'],
                                pd.to_datetime(drug_forecasts['forecast_date'])
                            ]),
                            'value': pd.concat([
                                history_pd[f'total_{metric}'],
                                drug_forecasts['forecast_value']
                            ]),
                            'type': ['historical'] * len(history_pd) + ['forecast'] * len(drug_forecasts)
                        })
                        
                        fig = px.line(
                            forecast_plot,
                            x="quarter_start",
                            y="value",
                            color="type",
                            title=f"Historical and Forecast {metric.title()}",
                            labels={"value": f"Total {metric.title()}", "quarter_start": "Quarter"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading processed data: {str(e)}")
        st.info("Please run the data processing pipeline again")


# Footer
st.markdown("---")
st.markdown("Medicaid Drug Spend Risk Forecasting Dashboard | Built with PySpark and Prophet")

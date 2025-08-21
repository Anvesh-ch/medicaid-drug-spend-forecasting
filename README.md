# Medicaid Drug Spend Risk Forecasting

A modular and lean project for processing Medicaid State Drug Utilization records, computing risk metrics, and forecasting drug spend trends using PySpark and Prophet.

## Overview

This project processes 5M+ Medicaid claims records to:
- Standardize data to a canonical schema
- Compute quarterly aggregates and trend features
- Assess risk levels based on cost, volatility, and growth patterns
- Forecast drug spend/units using Facebook Prophet
- Provide an interactive Streamlit dashboard for analysis

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py       # Package initialization
│   ├── io.py             # Spark session, CSV reading, Parquet I/O
│   ├── schema.py         # Column mapping and data validation
│   ├── transforms.py     # Data cleansing and quarterly aggregation
│   ├── features.py       # Trend features and rolling statistics
│   ├── risk.py           # Risk assessment rules and scoring
│   ├── forecast.py       # Prophet forecasting pipeline
│   └── sql.py            # SQL convenience functions and views
├── data/
│   ├── raw/              # Place CSV files here (gitignored)
│   └── processed/        # Parquet outputs (gitignored)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Data Requirements

Place your Medicaid State Drug Utilization CSV files in the `data/raw/` directory. The system will automatically detect and process all CSV files.

### Expected Columns

The system maps common column variations to canonical names:
- **Record ID** → `record_type` (FFSU or MCOU)
- **State Code** → `state` (2-letter uppercase)
- **NDC** → `ndc11` (11-digit normalized)
- **Year** → `year`
- **Quarter** → `quarter`
- **Units Reimbursed** → `units_reimbursed`
- **Total Amount Reimbursed** → `total_amount`
- **Medicaid Amount Reimbursed** → `medicaid_amount`
- And more...

## Quickstart

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate          # On macOS/Linux
# or
.venv\Scripts\activate             # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Create data directories
mkdir -p data/raw data/processed

# Place your CSV files in data/raw/
# Example: data/raw/SDUD-2023.csv
```

### 3. Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Usage

### Data Processing Pipeline

1. **Click "Process Data Pipeline"** in the sidebar to:
   - Read and validate CSV files
   - Apply canonical schema
   - Compute quarterly aggregates
   - Calculate trend features
   - Assess risk levels
   - Create SQL views

2. **Run Forecasts** to:
   - Select top K drugs per state
   - Run Prophet forecasts for next 4 quarters
   - Save results to Parquet

### Dashboard Features

#### Overview
- **KPI Tiles**: Total spend, unique drugs, states, latest quarter
- **Top Drugs**: Bar charts of highest spend and fastest risers
- **Risk Distribution**: Pie chart of risk levels

#### Risk Watchlist
- **High/Medium Risk Drugs**: Table with risk scores and flags
- **Download CSV**: Export current risk watchlist
- **Risk Summary**: Distribution by state and level

#### Trends by Drug
- **State/NDC Selection**: Interactive dropdowns
- **Historical Trends**: Line charts for spend and units
- **Forecasts**: Prophet predictions with confidence intervals

### Controls

- **State Filter**: Multi-select states for analysis
- **Date Range**: Filter by quarter range
- **Forecast Metric**: Choose between spend or units
- **Top K**: Adjust number of drugs per state for forecasting

## Performance Notes

- **Spark Configuration**: Optimized for laptop use with 4GB driver memory
- **Partitioning**: Data repartitioned by state for efficient joins
- **Caching**: Intermediate results saved to Parquet for reuse
- **Forecasting**: Limited universe selection to keep processing time reasonable

## Risk Assessment Rules

### Risk Levels
- **High**: High cost + (high volatility OR fast riser)
- **Medium**: High cost OR high volatility OR fast riser
- **Low**: All other cases

### Risk Flags
- **High Cost**: ≥75th percentile spend within state/quarter
- **High Volatility**: CV > 1.0 or QoQ change > 35%
- **Fast Riser**: 4-quarter growth > 25%
- **Extreme Volatility**: CV > 2.0 or QoQ change > 75%

## Technical Details

### Data Flow
1. **Raw CSV** → Schema detection and mapping
2. **Validation** → Data quality checks and cleansing
3. **Aggregation** → Quarterly rollups by state/NDC
4. **Features** → Trend analysis and rolling statistics
5. **Risk** → Threshold-based flagging and scoring
6. **Forecasting** → Prophet models for selected universe
7. **Dashboard** → Interactive visualization and analysis

### Key Technologies
- **PySpark**: Data processing and SQL queries
- **Prophet**: Time series forecasting
- **Streamlit**: Interactive web dashboard
- **Plotly**: Data visualization
- **Parquet**: Efficient data storage

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `top_k` in forecasting or increase Spark driver memory
2. **Slow Processing**: Check if data/processed/ contains cached results
3. **Missing Columns**: Verify CSV structure matches expected format
4. **Forecast Failures**: Ensure sufficient historical data (minimum 4 quarters)

### Performance Tuning

- Adjust `spark.driver.memory` in `src/io.py` for your system
- Modify `max_workers` in forecasting for parallel processing
- Use smaller `top_k` values for faster forecasting

## Data Dictionary

### Canonical Schema
- `record_type`: Record identifier (FFSU/MCOU)
- `state`: 2-letter state code
- `ndc11_normalized`: 11-digit NDC code
- `quarter_start`: First day of quarter
- `total_spend`: Total reimbursement amount
- `total_units`: Total units reimbursed
- `total_scripts`: Total prescriptions
- `risk_level`: Calculated risk category
- `risk_score`: Numerical risk score (0-100)

### Derived Metrics
- `qoq_spend_change`: Quarter-over-quarter spend change
- `cv_spend`: Coefficient of variation for spend
- `rolling_4q_growth_spend`: 4-quarter rolling growth rate
- `spend_per_script`: Average spend per prescription
- `spend_per_unit`: Average spend per unit

## Contributing

This project follows a modular design pattern. To extend functionality:

1. **Add new transforms** in `src/transforms.py`
2. **Extend risk rules** in `src/risk.py`
3. **Enhance features** in `src/features.py`
4. **Modify dashboard** in `app.py`

## License

This project is designed for educational and research purposes. Please ensure compliance with data usage agreements for Medicaid data.

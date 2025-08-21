#!/usr/bin/env python3
"""
Example usage script for the Medicaid Drug Spend Risk Forecasting project.
This script demonstrates how to use the modules programmatically.
"""

import os
from src.io import get_spark, read_raw_csvs, write_parquet
from src.schema import detect_columns, apply_schema, validate_ranges
from src.transforms import quarterly_aggregates, cleanse_data, add_time_dimensions
from src.features import trend_features
from src.risk import flag_risks
from src.forecast import select_forecast_universe


def main():
    """Main example function."""
    print("Medicaid Drug Spend Risk Forecasting - Example Usage")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists("data/raw"):
        print("❌ data/raw/ directory not found.")
        print("Please create data/raw/ and place your CSV files there.")
        return
    
    # Check for CSV files
    csv_files = [f for f in os.listdir("data/raw") if f.endswith('.csv')]
    if not csv_files:
        print("❌ No CSV files found in data/raw/")
        print("Please place your Medicaid CSV files in data/raw/")
        return
    
    print(f"✓ Found {len(csv_files)} CSV file(s): {csv_files}")
    
    try:
        # Initialize Spark
        print("\n1. Initializing Spark session...")
        spark = get_spark("medicaid-example")
        print("✓ Spark session initialized")
        
        # Read raw data
        print("\n2. Reading raw CSV files...")
        raw_df = read_raw_csvs(spark, "data/raw")
        print(f"✓ Loaded {raw_df.count():,} records")
        
        # Detect and apply schema
        print("\n3. Detecting and applying schema...")
        column_mapping = detect_columns(raw_df)
        print(f"✓ Detected {len(column_mapping)} columns")
        
        schema_df = apply_schema(raw_df, column_mapping)
        validated_df = validate_ranges(schema_df)
        print("✓ Schema applied and validation completed")
        
        # Cleanse data
        print("\n4. Cleansing data...")
        cleansed_df = cleanse_data(validated_df)
        print(f"✓ Data cleansing completed, {cleansed_df.count():,} records remaining")
        
        # Compute aggregates
        print("\n5. Computing quarterly aggregates...")
        aggregated_df = quarterly_aggregates(cleansed_df)
        print("✓ Quarterly aggregates computed")
        
        # Calculate features
        print("\n6. Calculating trend features...")
        features_df = trend_features(aggregated_df)
        print("✓ Trend features calculated")
        
        # Assess risks
        print("\n7. Assessing risks...")
        risk_df = flag_risks(features_df, by_state=True)
        print("✓ Risk assessment completed")
        
        # Select forecast universe
        print("\n8. Selecting forecast universe...")
        universe = select_forecast_universe(features_df, top_k_per_state=10, metric="spend")
        print(f"✓ Selected {universe.count():,} drugs for forecasting")
        
        # Save results
        print("\n9. Saving results...")
        os.makedirs("data/processed", exist_ok=True)
        write_parquet(aggregated_df, "data/processed/aggregates.parquet")
        write_parquet(features_df, "data/processed/features.parquet")
        write_parquet(risk_df, "data/processed/risks.parquet")
        print("✓ Results saved to data/processed/")
        
        print("\n🎉 Example pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Run 'streamlit run app.py' to start the dashboard")
        print("2. Use the dashboard to explore results and run forecasts")
        
    except Exception as e:
        print(f"\n❌ Error in example pipeline: {str(e)}")
        print("Please check your data format and try again.")
    
    finally:
        if 'spark' in locals():
            spark.stop()
            print("\n✓ Spark session stopped")


if __name__ == "__main__":
    main()

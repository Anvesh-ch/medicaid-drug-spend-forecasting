from pyspark.sql import DataFrame
from pyspark.sql.functions import col, sum as spark_sum, count, when, isnull, lit, nullif, concat, year, quarter
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType


def quarterly_aggregates(df: DataFrame) -> DataFrame:
    """Compute quarterly aggregates grouped by state, ndc11, product_name_fda10, quarter_start."""
    
    # Fill null values with 0 for aggregation
    filled_df = df.fillna({'units_reimbursed': 0, 'prescriptions': 0, 'total_amount': 0, 'medicaid_amount': 0, 'non_medicaid_amount': 0})
    
    # Group by dimensions and aggregate
    aggregated = filled_df.groupBy(
        'state', 'ndc11_normalized', 'product_name_fda10', 'quarter_start'
    ).agg(
        spark_sum('units_reimbursed').alias('total_units'),
        spark_sum('prescriptions').alias('total_scripts'),
        spark_sum('total_amount').alias('total_spend'),
        spark_sum('medicaid_amount').alias('medicaid_spend'),
        spark_sum('non_medicaid_amount').alias('non_medicaid_spend')
    )
    
    # Calculate derived metrics with safe division
    with_metrics = aggregated.withColumn(
        'spend_per_script',
        when(col('total_scripts') > 0, 
             col('total_spend') / col('total_scripts'))
        .otherwise(lit(None))
    ).withColumn(
        'spend_per_unit',
        when(col('total_units') > 0, 
             col('total_spend') / col('total_units'))
        .otherwise(lit(None))
    ).withColumn(
        'medicaid_share',
        when(col('total_spend') > 0, 
             col('medicaid_spend') / col('total_spend'))
        .otherwise(lit(None))
    )
    
    # Repartition by state for better downstream performance
    repartitioned = with_metrics.repartition('state')
    
    return repartitioned


def cleanse_data(df: DataFrame) -> DataFrame:
    """Apply data cleansing rules."""
    cleansed = df
    
    # Remove records with invalid quarters
    if 'invalid_quarter' in df.columns:
        cleansed = cleansed.filter(col('invalid_quarter') == False)
    
    # Remove records with invalid states
    if 'invalid_state' in df.columns:
        cleansed = cleansed.filter(col('invalid_state') == False)
    
    # Remove records with invalid NDCs
    if 'invalid_ndc' in df.columns:
        cleansed = cleansed.filter(col('invalid_ndc') == False)
    
    # Remove records with missing critical fields
    cleansed = cleansed.filter(
        col('state').isNotNull() &
        col('ndc11_normalized').isNotNull() &
        col('quarter_start').isNotNull()
    )
    
    # Ensure numeric fields are non-negative
    numeric_cols = ['total_units', 'total_scripts', 'total_spend']
    for col_name in numeric_cols:
        if col_name in cleansed.columns:
            cleansed = cleansed.filter(col(col_name) >= 0)
    
    return cleansed


def add_time_dimensions(df: DataFrame) -> DataFrame:
    """Add time-based dimensions for analysis."""
    # Skip adding time dimensions as they're already available
    return df

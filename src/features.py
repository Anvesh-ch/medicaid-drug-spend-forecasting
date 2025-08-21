from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lag, lead, avg, stddev, when, lit, sum as spark_sum
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType


def trend_features(df: DataFrame) -> DataFrame:
    """Calculate trend features using windowed functions."""
    
    # Define window for state and NDC level analysis
    window_spec = Window.partitionBy('state', 'ndc11_normalized').orderBy('quarter_start')
    
    # Window for rolling 4-quarter calculations
    rolling_4q = Window.partitionBy('state', 'ndc11_normalized').orderBy('quarter_start').rowsBetween(-3, 0)
    rolling_8q = Window.partitionBy('state', 'ndc11_normalized').orderBy('quarter_start').rowsBetween(-7, -4)
    
    # Calculate quarter-over-quarter changes (simplified)
    with_qoq = df.withColumn(
        'qoq_spend_change',
        when((col('total_spend').isNotNull()) & 
             (lag('total_spend', 1).over(window_spec).isNotNull()) &
             (lag('total_spend', 1).over(window_spec) != 0),
             (col('total_spend') - lag('total_spend', 1).over(window_spec)) / 
             lag('total_spend', 1).over(window_spec))
        .otherwise(lit(None))
    ).withColumn(
        'qoq_units_change',
        when((col('total_units').isNotNull()) & 
             (lag('total_units', 1).over(window_spec).isNotNull()) &
             (lag('total_units', 1).over(window_spec) != 0),
             (col('total_units') - lag('total_units', 1).over(window_spec)) / 
             lag('total_units', 1).over(window_spec))
        .otherwise(lit(None))
    ).withColumn(
        'qoq_scripts_change',
        when((col('total_scripts').isNotNull()) & 
             (lag('total_scripts', 1).over(window_spec).isNotNull()) &
             (lag('total_scripts', 1).over(window_spec) != 0),
             (col('total_scripts') - lag('total_scripts', 1).over(window_spec)) / 
             lag('total_scripts', 1).over(window_spec))
        .otherwise(lit(None))
    )
    
    # Calculate rolling 4-quarter statistics
    with_rolling = with_qoq.withColumn(
        't4q_spend_avg',
        avg('total_spend').over(rolling_4q)
    ).withColumn(
        't4q_spend_std',
        stddev('total_spend').over(rolling_4q)
    ).withColumn(
        't4q_units_avg',
        avg('total_units').over(rolling_4q)
    ).withColumn(
        't4q_units_std',
        stddev('total_units').over(rolling_4q)
    ).withColumn(
        't4q_scripts_avg',
        avg('total_scripts').over(rolling_4q)
    ).withColumn(
        't4q_scripts_std',
        stddev('total_scripts').over(rolling_4q)
    )
    
    # Calculate coefficient of variation (simplified)
    with_cv = with_rolling.withColumn(
        'cv_spend',
        when((col('t4q_spend_avg') > 0) & (col('t4q_spend_std').isNotNull()),
             col('t4q_spend_std') / col('t4q_spend_avg'))
        .otherwise(lit(None))
    ).withColumn(
        'cv_units',
        when((col('t4q_units_avg') > 0) & (col('t4q_units_std').isNotNull()),
             col('t4q_units_std') / col('t4q_units_avg'))
        .otherwise(lit(None))
    ).withColumn(
        'cv_scripts',
        when((col('t4q_scripts_avg') > 0) & (col('t4q_scripts_std').isNotNull()),
             col('t4q_scripts_std') / col('t4q_scripts_avg'))
        .otherwise(lit(None))
    )
    
    # Calculate rolling 4-quarter growth rates (simplified)
    with_growth = with_cv.withColumn(
        'rolling_4q_growth_spend',
        when(
            (spark_sum('total_spend').over(rolling_4q).isNotNull()) & 
            (spark_sum('total_spend').over(rolling_8q).isNotNull()) &
            (spark_sum('total_spend').over(rolling_8q) > 0),
            (spark_sum('total_spend').over(rolling_4q) - spark_sum('total_spend').over(rolling_8q)) / 
            spark_sum('total_spend').over(rolling_8q)
        ).otherwise(lit(None))
    ).withColumn(
        'rolling_4q_growth_units',
        when(
            (spark_sum('total_units').over(rolling_4q).isNotNull()) & 
            (spark_sum('total_units').over(rolling_8q).isNotNull()) &
            (spark_sum('total_units').over(rolling_8q) > 0),
            (spark_sum('total_units').over(rolling_4q) - spark_sum('total_units').over(rolling_8q)) / 
            spark_sum('total_units').over(rolling_8q)
        ).otherwise(lit(None))
    )
    
    # Add trend direction indicators
    with_trends = with_growth.withColumn(
        'trend_direction_spend',
        when(col('qoq_spend_change') > 0.05, lit('increasing'))
        .when(col('qoq_spend_change') < -0.05, lit('decreasing'))
        .otherwise(lit('stable'))
    ).withColumn(
        'trend_direction_units',
        when(col('qoq_units_change') > 0.05, lit('increasing'))
        .when(col('qoq_units_change') < -0.05, lit('decreasing'))
        .otherwise(lit('stable'))
    )
    
    return with_trends


def add_lag_features(df: DataFrame, lags: list = [1, 2, 4]) -> DataFrame:
    """Add lagged features for time series analysis."""
    window_spec = Window.partitionBy('state', 'ndc11_normalized').orderBy('quarter_start')
    
    with_lags = df
    
    for lag_period in lags:
        with_lags = with_lags.withColumn(
            f'spend_lag_{lag_period}q',
            lag('total_spend', lag_period).over(window_spec)
        ).withColumn(
            f'units_lag_{lag_period}q',
            lag('total_units', lag_period).over(window_spec)
        ).withColumn(
            f'scripts_lag_{lag_period}q',
            lag('total_scripts', lag_period).over(window_spec)
        )
    
    return with_lags
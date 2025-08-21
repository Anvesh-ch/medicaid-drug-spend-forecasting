from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit, percentile_approx, expr, abs
from pyspark.sql.window import Window


def flag_risks(df: DataFrame, by_state: bool = True) -> DataFrame:
    """Flag risks based on thresholds and rules."""
    
    if by_state:
        # Calculate state-level percentiles for high cost threshold
        window_spec = Window.partitionBy('state', 'quarter_start')
        
        with_percentiles = df.withColumn(
            'spend_75th_percentile',
            percentile_approx('total_spend', 0.75).over(window_spec)
        ).withColumn(
            'spend_90th_percentile',
            percentile_approx('total_spend', 0.90).over(window_spec)
        )
    else:
        # Global percentiles
        with_percentiles = df.withColumn(
            'spend_75th_percentile',
            percentile_approx('total_spend', 0.75)
        ).withColumn(
            'spend_90th_percentile',
            percentile_approx('total_spend', 0.90)
        )
    
    # Flag high cost drugs
    with_high_cost = with_percentiles.withColumn(
        'high_cost',
        when(col('total_spend') >= col('spend_75th_percentile'), lit(True))
        .otherwise(lit(False))
    ).withColumn(
        'very_high_cost',
        when(col('total_spend') >= col('spend_90th_percentile'), lit(True))
        .otherwise(lit(False))
    )
    
    # Flag high volatility drugs
    with_volatility = with_high_cost.withColumn(
        'high_vol',
        when(
            (col('cv_spend') > 1.0) | 
            (abs(col('qoq_spend_change')) > 0.35) |
            (col('cv_spend').isNull() & (abs(col('qoq_spend_change')) > 0.50)),
            lit(True)
        ).otherwise(lit(False))
    ).withColumn(
        'extreme_vol',
        when(
            (col('cv_spend') > 2.0) | 
            (abs(col('qoq_spend_change')) > 0.75),
            lit(True)
        ).otherwise(lit(False))
    )
    
    # Flag fast risers
    with_risers = with_volatility.withColumn(
        'fast_riser',
        when(
            col('rolling_4q_growth_spend') > 0.25,
            lit(True)
        ).otherwise(lit(False))
    ).withColumn(
        'extreme_riser',
        when(
            col('rolling_4q_growth_spend') > 0.50,
            lit(True)
        ).otherwise(lit(False))
    )
    
    # Flag declining drugs
    with_declining = with_risers.withColumn(
        'declining',
        when(
            col('rolling_4q_growth_spend') < -0.25,
            lit(True)
        ).otherwise(lit(False))
    ).withColumn(
        'rapid_decline',
        when(
            col('rolling_4q_growth_spend') < -0.50,
            lit(True)
        ).otherwise(lit(False))
    )
    
    # Calculate risk level
    with_risk_level = with_declining.withColumn(
        'risk_level',
        when(
            col('high_cost') & (col('high_vol') | col('fast_riser') | col('extreme_riser')),
            lit('high')
        ).when(
            col('high_cost') | col('high_vol') | col('fast_riser') | col('extreme_riser') |
            col('extreme_vol') | col('rapid_decline'),
            lit('medium')
        ).otherwise(lit('low'))
    )
    
    # Add risk score (0-100)
    with_risk_score = with_risk_level.withColumn(
        'risk_score',
        when(col('risk_level') == 'high', lit(100))
        .when(col('risk_level') == 'medium', lit(50))
        .otherwise(lit(0))
    ).withColumn(
        'risk_score',
        when(col('high_cost'), col('risk_score') + 20).otherwise(col('risk_score'))
    ).withColumn(
        'risk_score',
        when(col('high_vol'), col('risk_score') + 15).otherwise(col('risk_score'))
    ).withColumn(
        'risk_score',
        when(col('fast_riser'), col('risk_score') + 15).otherwise(col('risk_score'))
    ).withColumn(
        'risk_score',
        when(col('extreme_vol'), col('risk_score') + 10).otherwise(col('risk_score'))
    ).withColumn(
        'risk_score',
        when(col('extreme_riser'), col('risk_score') + 10).otherwise(col('risk_score'))
    ).withColumn(
        'risk_score',
        when(col('rapid_decline'), col('risk_score') + 10).otherwise(col('risk_score'))
    )
    
    # Ensure risk score doesn't exceed 100
    final_risk = with_risk_score.withColumn(
        'risk_score',
        when(col('risk_score') > 100, lit(100)).otherwise(col('risk_score'))
    )
    
    return final_risk


def get_latest_quarter_risks(df: DataFrame) -> DataFrame:
    """Get risk assessment for the latest available quarter per state-NDC combination."""
    window_spec = Window.partitionBy('state', 'ndc11_normalized').orderBy(col('quarter_start').desc())
    
    latest_risks = df.withColumn(
        'row_num',
        expr('row_number() over (partition by state, ndc11_normalized order by quarter_start desc)')
    ).filter(col('row_num') == 1).drop('row_num')
    
    return latest_risks


def summarize_risk_by_state(df: DataFrame) -> DataFrame:
    """Summarize risk distribution by state."""
    risk_summary = df.groupBy('state', 'risk_level').agg(
        expr('count(*) as drug_count'),
        expr('sum(total_spend) as total_spend_at_risk'),
        expr('avg(risk_score) as avg_risk_score')
    )
    
    return risk_summary

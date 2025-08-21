from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, desc, row_number, expr, sum as spark_sum, avg, count
from pyspark.sql.window import Window


def create_temp_views(spark: SparkSession, df_agg: DataFrame, df_feat: DataFrame, df_risk: DataFrame):
    """Create temporary views for SQL queries."""
    
    # Create temp views
    df_agg.createOrReplaceTempView("aggregates")
    df_feat.createOrReplaceTempView("features")
    df_risk.createOrReplaceTempView("risks")
    
    # Create combined view for easier querying
    combined = df_agg.join(
        df_feat, 
        on=['state', 'ndc11_normalized', 'quarter_start'], 
        how='inner'
    ).join(
        df_risk, 
        on=['state', 'ndc11_normalized', 'quarter_start'], 
        how='inner'
    )
    
    combined.createOrReplaceTempView("combined_data")


def get_top_spend_by_state_latest(spark: SparkSession, state_filter: str = None, limit: int = 10) -> DataFrame:
    """Get top spend drugs by state for latest quarter."""
    
    query = """
    SELECT 
        state,
        ndc11_normalized,
        product_name_fda10,
        total_spend,
        total_units,
        total_scripts,
        quarter_start,
        risk_level,
        risk_score
    FROM combined_data c1
    WHERE quarter_start = (
        SELECT MAX(quarter_start) 
        FROM combined_data c2 
        WHERE c2.state = c1.state
    )
    """
    
    if state_filter:
        query += f" AND state = '{state_filter}'"
    
    query += f" ORDER BY total_spend DESC LIMIT {limit}"
    
    return spark.sql(query)


def get_top_risers_by_state_latest(spark: SparkSession, state_filter: str = None, limit: int = 10) -> DataFrame:
    """Get top risers by quarter-over-quarter change for latest quarter."""
    
    query = """
    SELECT 
        state,
        ndc11_normalized,
        product_name_fda10,
        total_spend,
        qoq_spend_change,
        rolling_4q_growth_spend,
        cv_spend,
        risk_level,
        risk_score,
        quarter_start
    FROM combined_data c1
    WHERE quarter_start = (
        SELECT MAX(quarter_start) 
        FROM combined_data c2 
        WHERE c2.state = c1.state
    )
    AND qoq_spend_change IS NOT NULL
    """
    
    if state_filter:
        query += f" AND state = '{state_filter}'"
    
    query += f" ORDER BY qoq_spend_change DESC LIMIT {limit}"
    
    return spark.sql(query)


def get_history_by_state_ndc(spark: SparkSession, state: str, ndc11: str) -> DataFrame:
    """Get historical data for a specific state and NDC."""
    
    query = f"""
    SELECT 
        quarter_start,
        total_spend,
        total_units,
        total_scripts,
        qoq_spend_change,
        rolling_4q_growth_spend,
        cv_spend,
        risk_level,
        risk_score
    FROM combined_data
    WHERE state = '{state}' AND ndc11_normalized = '{ndc11}'
    ORDER BY quarter_start
    """
    
    return spark.sql(query)


def get_risk_summary_by_state(spark: SparkSession) -> DataFrame:
    """Get risk summary by state."""
    
    query = """
    SELECT 
        state,
        risk_level,
        COUNT(*) as drug_count,
        SUM(total_spend) as total_spend_at_risk,
        AVG(risk_score) as avg_risk_score,
        MAX(quarter_start) as latest_quarter
    FROM combined_data
    GROUP BY state, risk_level
    ORDER BY state, 
             CASE risk_level 
                 WHEN 'high' THEN 1 
                 WHEN 'medium' THEN 2 
                 WHEN 'low' THEN 3 
             END
    """
    
    return spark.sql(query)


def get_high_risk_drugs(spark: SparkSession, state_filter: str = None, limit: int = 20) -> DataFrame:
    """Get high and medium risk drugs."""
    
    query = """
    SELECT 
        state,
        ndc11_normalized,
        product_name_fda10,
        total_spend,
        qoq_spend_change,
        rolling_4q_growth_spend,
        cv_spend,
        risk_level,
        risk_score,
        quarter_start
    FROM combined_data
    WHERE risk_level IN ('high', 'medium')
    """
    
    if state_filter:
        query += f" AND state = '{state_filter}'"
    
    query += f" ORDER BY risk_score DESC, total_spend DESC LIMIT {limit}"
    
    return spark.sql(query)


def get_spend_trends_by_state(spark: SparkSession, state_filter: str = None) -> DataFrame:
    """Get spend trends by state over time."""
    
    query = """
    SELECT 
        state,
        quarter_start,
        SUM(total_spend) as state_total_spend,
        COUNT(DISTINCT ndc11_normalized) as unique_drugs,
        AVG(risk_score) as avg_risk_score
    FROM combined_data
    """
    
    if state_filter:
        query += f" WHERE state = '{state_filter}'"
    
    query += """
    GROUP BY state, quarter_start
    ORDER BY state, quarter_start
    """
    
    return spark.sql(query)


def get_drug_universe_summary(spark: SparkSession) -> DataFrame:
    """Get summary statistics for the drug universe."""
    
    query = """
    SELECT 
        COUNT(DISTINCT state) as total_states,
        COUNT(DISTINCT ndc11_normalized) as total_drugs,
        COUNT(DISTINCT CONCAT(state, ndc11_normalized)) as total_state_drug_combinations,
        MIN(quarter_start) as earliest_quarter,
        MAX(quarter_start) as latest_quarter,
        SUM(total_spend) as total_spend_all_time
    FROM combined_data
    """
    
    return spark.sql(query)

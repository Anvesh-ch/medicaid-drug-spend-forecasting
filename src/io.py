import os
from functools import wraps
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType


def get_spark(app_name="medicaid-risk") -> SparkSession:
    """Create and configure Spark session for local development."""
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
            .master("local[*]")
            .getOrCreate())


def read_raw_csvs(spark: SparkSession, raw_dir: str = "data/raw") -> DataFrame:
    """Read all CSV files from raw directory with proper options."""
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory {raw_dir} not found")
    
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    
    # Read first CSV to get schema
    first_file = os.path.join(raw_dir, csv_files[0])
    sample_df = spark.read.csv(first_file, header=True, inferSchema=True)
    
    # Read all CSVs with same schema
    all_dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(raw_dir, csv_file)
        df = spark.read.csv(file_path, header=True, schema=sample_df.schema)
        all_dfs.append(df)
    
    # Union all dataframes
    if len(all_dfs) == 1:
        return all_dfs[0]
    else:
        return all_dfs[0].unionByName(*all_dfs[1:], allowMissingColumns=True)


def write_parquet(df: DataFrame, path: str) -> None:
    """Write DataFrame to Parquet format with partitioning."""
    df.write.mode("overwrite").parquet(path)


def read_parquet(path: str) -> DataFrame:
    """Read DataFrame from Parquet format."""
    return SparkSession.getActiveSession().read.parquet(path)


def cache_if_absent(fn):
    """Decorator to cache function results if output file doesn't exist."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Extract output path from kwargs if available
        output_path = kwargs.get('output_path')
        if output_path and os.path.exists(output_path):
            return read_parquet(output_path)
        return fn(*args, **kwargs)
    return wrapper

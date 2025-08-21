import re
from datetime import date
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, trim, upper, when, isnan, isnull, lit, length, abs, regexp_replace, lpad, to_date, concat_ws
from pyspark.sql.types import StringType, IntegerType, DoubleType, LongType, DateType, BooleanType


def detect_columns(df: DataFrame) -> dict:
    """Detect actual column names and map to canonical schema."""
    actual_cols = df.columns
    mapping = {}
    
    # Common column name variations
    column_variations = {
        'record_type': ['Utilization Type', 'Record ID', 'RecordID', 'record_id'],
        'state': ['State', 'State Code', 'StateCode', 'state_code'],
        'ndc11': ['NDC', 'ndc', 'National Drug Code'],
        'ndc_labeler': ['Labeler Code', 'LabelerCode', 'labeler_code'],
        'ndc_product': ['Product Code', 'ProductCode', 'product_code'],
        'ndc_package': ['Package Size', 'PackageSize', 'package_size'],
        'year': ['Year', 'year'],
        'quarter': ['Quarter', 'quarter', 'Qtr'],
        'suppression_used': ['Suppression Used', 'SuppressionUsed', 'suppression_used'],
        'product_name_fda10': ['Product Name', 'Product FDA List Name', 'ProductFDAListName', 'product_name'],
        'units_reimbursed': ['Units Reimbursed', 'UnitsReimbursed', 'units_reimbursed'],
        'prescriptions': ['Number of Prescriptions', 'No. of Prescriptions', 'NoOfPrescriptions', 'prescriptions', 'Scripts'],
        'total_amount': ['Total Amount Reimbursed', 'TotalAmountReimbursed', 'total_amount'],
        'medicaid_amount': ['Medicaid Amount Reimbursed', 'MedicaidAmountReimbursed', 'medicaid_amount'],
        'non_medicaid_amount': ['Non Medicaid Amount Reimbursed', 'Non-Medicaid Amount Reimbursed', 'NonMedicaidAmountReimbursed', 'non_medicaid_amount']
    }
    
    for canonical, variations in column_variations.items():
        for variation in variations:
            if variation in actual_cols:
                mapping[canonical] = variation
                break
    
    return mapping


def apply_schema(df: DataFrame, mapping: dict) -> DataFrame:
    """Apply canonical schema with proper data types and validation."""
    # Start with original columns and apply transformations
    transformed_df = df
    
    # Apply column mappings and transformations
    select_exprs = []
    
    for canonical, actual in mapping.items():
        if canonical == 'record_type':
            select_exprs.append(trim(col(actual)).alias(canonical))
        elif canonical == 'state':
            select_exprs.append(upper(trim(col(actual))).alias(canonical))
        elif canonical == 'ndc11':
            select_exprs.append(trim(col(actual)).alias(canonical))
        elif canonical == 'ndc_labeler':
            select_exprs.append(trim(col(actual)).alias(canonical))
        elif canonical == 'ndc_product':
            select_exprs.append(trim(col(actual)).alias(canonical))
        elif canonical == 'ndc_package':
            select_exprs.append(trim(col(actual)).alias(canonical))
        elif canonical == 'year':
            select_exprs.append(col(actual).cast(IntegerType()).alias(canonical))
        elif canonical == 'quarter':
            select_exprs.append(col(actual).cast(IntegerType()).alias(canonical))
        elif canonical == 'suppression_used':
            select_exprs.append(trim(col(actual)).alias(canonical))
        elif canonical == 'product_name_fda10':
            select_exprs.append(trim(col(actual)).alias(canonical))
        elif canonical == 'units_reimbursed':
            select_exprs.append(col(actual).cast(DoubleType()).alias(canonical))
        elif canonical == 'prescriptions':
            select_exprs.append(col(actual).cast(LongType()).alias(canonical))
        elif canonical == 'total_amount':
            select_exprs.append(col(actual).cast(DoubleType()).alias(canonical))
        elif canonical == 'medicaid_amount':
            select_exprs.append(col(actual).cast(DoubleType()).alias(canonical))
        elif canonical == 'non_medicaid_amount':
            select_exprs.append(col(actual).cast(DoubleType()).alias(canonical))
    
    # Add derived columns using built-in functions
    # Add quarter_start if year and quarter exist
    if 'year' in mapping and 'quarter' in mapping:
        # Create quarter start date: YYYY-QQ-01 format, then convert to actual start month
        quarter_start_expr = (
            when(col('quarter') == 1, concat_ws('-', col('year'), lit('01'), lit('01')))
            .when(col('quarter') == 2, concat_ws('-', col('year'), lit('04'), lit('01')))
            .when(col('quarter') == 3, concat_ws('-', col('year'), lit('07'), lit('01')))
            .when(col('quarter') == 4, concat_ws('-', col('year'), lit('10'), lit('01')))
            .otherwise(lit(None))
        )
        select_exprs.append(to_date(quarter_start_expr, 'yyyy-MM-dd').alias('quarter_start'))
    
    # Add NDC normalization using built-in functions
    if 'ndc11' in mapping:
        # Remove all non-digits and pad to 11 digits
        ndc_normalized = lpad(regexp_replace(col('ndc11'), r'[^\d]', ''), 11, '0')
        select_exprs.append(ndc_normalized.alias('ndc11_normalized'))
    
    # Add spend mismatch flag
    if all(col_name in mapping for col_name in ['total_amount', 'medicaid_amount', 'non_medicaid_amount']):
        mismatch_condition = (
            col('total_amount').isNotNull() & 
            col('medicaid_amount').isNotNull() & 
            col('non_medicaid_amount').isNotNull() &
            (abs(col('total_amount') - (col('medicaid_amount') + col('non_medicaid_amount'))) > 0.01)
        )
        select_exprs.append(
            when(mismatch_condition, lit(True)).otherwise(lit(False)).alias('spend_mismatch')
        )
    
    # Select transformed columns
    transformed_df = transformed_df.select(*select_exprs)
    
    return transformed_df


def validate_ranges(df: DataFrame) -> DataFrame:
    """Add validation flags for data quality checks."""
    validated_df = df
    
    # Invalid quarter flag
    if 'quarter' in df.columns:
        validated_df = validated_df.withColumn(
            'invalid_quarter',
            when((col('quarter') < 1) | (col('quarter') > 4), lit(True))
            .otherwise(lit(False))
        )
    
    # Invalid state flag (check if 2 letters)
    if 'state' in df.columns:
        validated_df = validated_df.withColumn(
            'invalid_state',
            when(length(col('state')) != 2, lit(True))
            .otherwise(lit(False))
        )
    
    # Invalid NDC flag
    if 'ndc11_normalized' in df.columns:
        validated_df = validated_df.withColumn(
            'invalid_ndc',
            when((col('ndc11_normalized').isNull()) | 
                 (length(col('ndc11_normalized')) != 11) |
                 (~col('ndc11_normalized').rlike(r'^\d{11}$')), lit(True))
            .otherwise(lit(False))
        )
    
    # Handle negative values
    numeric_cols = ['units_reimbursed', 'prescriptions', 'total_amount', 'medicaid_amount', 'non_medicaid_amount']
    for col_name in numeric_cols:
        if col_name in df.columns:
            validated_df = validated_df.withColumn(
                col_name,
                when(col(col_name) < 0, lit(None)).otherwise(col(col_name))
            )
    
    return validated_df
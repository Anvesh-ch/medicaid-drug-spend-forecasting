import os
import pandas as pd
from typing import List, Dict, Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, desc, row_number, expr
from pyspark.sql.window import Window
from prophet import Prophet
import joblib
from tqdm import tqdm


def select_forecast_universe(df: DataFrame, top_k_per_state: int = 50, metric: str = "spend") -> DataFrame:
    """Select top K drugs per state for forecasting based on metric."""
    
    # Get latest quarter data
    window_spec = Window.partitionBy('state', 'ndc11_normalized').orderBy(col('quarter_start').desc())
    latest_data = df.withColumn(
        'row_num',
        row_number().over(window_spec)
    ).filter(col('row_num') == 1).drop('row_num')
    
    # Select top K by metric within each state
    ranking_window = Window.partitionBy('state').orderBy(desc(f'total_{metric}'))
    top_k_universe = latest_data.withColumn(
        'rank',
        row_number().over(ranking_window)
    ).filter(col('rank') <= top_k_per_state).drop('rank')
    
    return top_k_universe


def prepare_prophet_frame(df: DataFrame, state: str, ndc11: str, metric: str = "spend") -> pd.DataFrame:
    """Prepare Pandas DataFrame for Prophet forecasting."""
    
    # Filter for specific state and NDC
    filtered = df.filter(
        (col('state') == state) & 
        (col('ndc11_normalized') == ndc11)
    ).orderBy('quarter_start')
    
    # Convert to Pandas
    pandas_df = filtered.select('quarter_start', f'total_{metric}').toPandas()
    
    # Prophet expects 'ds' for dates and 'y' for values
    prophet_df = pandas_df.rename(columns={
        'quarter_start': 'ds',
        f'total_{metric}': 'y'
    })
    
    # Ensure date format and sort
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    # Remove any null values
    prophet_df = prophet_df.dropna()
    
    return prophet_df


def run_prophet_for_series(pd_df: pd.DataFrame, periods: int = 4) -> Dict:
    """Run Prophet forecast for a single time series."""
    
    if len(pd_df) < 4:  # Need at least 4 quarters for meaningful forecast
        return {
            'yhat': None,
            'yhat_lower': None,
            'yhat_upper': None,
            'forecast_dates': None,
            'error': 'Insufficient data'
        }
    
    try:
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.8
        )
        
        # Fit model
        model.fit(pd_df)
        
        # Create future dataframe for forecasting
        future = model.make_future_dataframe(periods=periods, freq='Q')
        
        # Make forecast
        forecast = model.predict(future)
        
        # Extract forecast values
        forecast_values = forecast.tail(periods)
        
        return {
            'yhat': forecast_values['yhat'].tolist(),
            'yhat_lower': forecast_values['yhat_lower'].tolist(),
            'yhat_upper': forecast_values['yhat_upper'].tolist(),
            'forecast_dates': forecast_values['ds'].tolist(),
            'error': None
        }
        
    except Exception as e:
        return {
            'yhat': None,
            'yhat_lower': None,
            'yhat_upper': None,
            'forecast_dates': None,
            'error': str(e)
        }


def run_batch_forecasts(df: DataFrame, universe: DataFrame, metric: str = "spend", 
                       max_workers: int = 4) -> pd.DataFrame:
    """Run forecasts for all drugs in the universe."""
    
    # Convert universe to Pandas for easier iteration
    universe_pd = universe.select('state', 'ndc11_normalized', 'product_name_fda10').toPandas()
    
    forecast_results = []
    
    # Process forecasts with progress bar
    for _, row in tqdm(universe_pd.iterrows(), total=len(universe_pd), desc="Running forecasts"):
        state = row['state']
        ndc11 = row['ndc11_normalized']
        product_name = row['product_name_fda10']
        
        # Prepare data for Prophet
        prophet_df = prepare_prophet_frame(df, state, ndc11, metric)
        
        # Run forecast
        forecast = run_prophet_for_series(prophet_df)
        
        # Store results
        if forecast['error'] is None:
            for i, (yhat, yhat_lower, yhat_upper, forecast_date) in enumerate(
                zip(forecast['yhat'], forecast['yhat_lower'], 
                    forecast['yhat_upper'], forecast['forecast_dates'])
            ):
                forecast_results.append({
                    'state': state,
                    'ndc11': ndc11,
                    'product_name': product_name,
                    'metric': metric,
                    'forecast_period': i + 1,
                    'forecast_date': forecast_date,
                    'forecast_value': yhat,
                    'forecast_lower': yhat_lower,
                    'forecast_upper': yhat_upper
                })
        else:
            # Log error case
            forecast_results.append({
                'state': state,
                'ndc11': ndc11,
                'product_name': product_name,
                'metric': metric,
                'forecast_period': None,
                'forecast_date': None,
                'forecast_value': None,
                'forecast_lower': None,
                'forecast_upper': None,
                'error': forecast['error']
            })
    
    # Convert to DataFrame
    forecasts_df = pd.DataFrame(forecast_results)
    
    return forecasts_df


def save_forecasts(forecasts_df: pd.DataFrame, output_path: str = "data/processed/forecasts.parquet"):
    """Save forecasts to Parquet format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecasts_df.to_parquet(output_path, index=False)


def load_forecasts(input_path: str = "data/processed/forecasts.parquet") -> pd.DataFrame:
    """Load forecasts from Parquet format."""
    if os.path.exists(input_path):
        return pd.read_parquet(input_path)
    else:
        return pd.DataFrame()

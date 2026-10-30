import pandas as pd

def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df: Input DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
        timeframe: target timeframe alias (e.g., '1H', '4H', '1D', 'W').
        
    Returns:
        pd.DataFrame: Resampled OHLCV data.
    """
    if df.empty:
        return df

    # logic provides a mapping for aggregation
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Handle columns that might be missing or different case
    # Basic normalization if needed, but assuming standard yfinance output
    # yfinance multi-index columns might need flattening before this if multiple tickers
    
    try:
        resampled = df.resample(timeframe).agg(agg_dict)
        # Drop rows with NaNs created by resampling (e.g. weekends in standard time)
        resampled.dropna(inplace=True) 
        return resampled
    except Exception as e:
        print(f"Resampling error: {e}")
        return pd.DataFrame()

def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters data to include only US Market hours (09:30 - 16:00 EST).
    Useful for aligning 24/7 assets (Crypto) with Stocks.
    """
    # Assuming the index is already timezone-aware or in UTC
    # Standard NYSE hours in UTC are roughly 14:30 - 21:00 (Standard) or 13:30 - 20:00 (Daylight)
    # For simplicity, we use local time check if timezone is set, or just the hours.
    
    # We use .between_time for convenience
    return df.between_time('14:30', '21:00') 


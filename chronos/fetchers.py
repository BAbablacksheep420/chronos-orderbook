import yfinance as yf
import pandas as pd
from typing import Optional, List, Union
from datetime import datetime, timedelta

def fetch_data(
    ticker: str, 
    period: str = "1y", 
    interval: str = "1d", 
    start: Optional[str] = None, 
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical data for a given ticker or list of tickers.
    
    Args:
        ticker: Symbol name (e.g., 'AAPL', 'BTC-USD').
        period: Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
        interval: Data resolution (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo).
        start: Start date string (YYYY-MM-DD). If provided, overrides period.
        end: End date string (YYYY-MM-DD).
    
    Returns:
        pd.DataFrame: OHLCV data.
    """
    print(f"Fetching {ticker} (Interval: {interval}, Period: {period})...")
    
    # yfinance uses 'auto_adjust' to handle splits/dividends
    try:
        df = yf.download(
            tickers=ticker, 
            period=period, 
            interval=interval, 
            start=start, 
            end=end,
            auto_adjust=True,
            progress=False
        )
        
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            return df
            
        # Flatten MultiIndex columns if present (yfinance > 0.2 behavior)
        if isinstance(df.columns, pd.MultiIndex):
            # Assuming format is (Price, Ticker), we want just Price
            # But only if we have single level of tickers or they are redundant
            # Since we usually fetch single ticker here, dropping level 1 is getting us back to [Open, High...]
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception as e:
                print(f"Error flattening columns: {e}")
                
        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def save_to_csv(df: pd.DataFrame, filename: str):
    """Save dataframe to CSV."""
    df.to_csv(filename)
    print(f"Saved to {filename}")

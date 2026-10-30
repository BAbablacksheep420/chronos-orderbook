import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

def rolling_correlation(
    series1: pd.Series, 
    series2: pd.Series, 
    window: int = 20, 
    method: str = 'pearson'
) -> pd.Series:
    """
    Calculate rolling correlation between two time series.
    """
    return series1.rolling(window=window).corr(series2)

def check_cointegration(series1: pd.Series, series2: pd.Series):
    """
    Perform Engle-Granger 2-step cointegration test.
    Returns p-value. H0: No cointegration.
    Low p-value (<0.05) implies cointegration.
    """
    # Simple proxies: Regression Residuals Stationarity
    # Step 1: Regress series2 on series1
    # Note: Requires simple linear regression, can use stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(series1, series2)
    residuals = series2 - (slope * series1 + intercept)
    
    # Step 2: Check stationarity of residuals (ADF Test)
    result = adfuller(residuals)
    return result[1] # p-value

def multi_timeframe_correlation(
    df1: pd.Series, 
    df2: pd.Series, 
    lags: list = [0, 1, 5, 20]
) -> dict:
    """
    Get correlation at different lags (Lagged Correlation).
    """
    correlations = {}
    for lag in lags:
        # Shift df2 by lag
        shifted_series2 = df2.shift(-lag)
        corr = df1.corr(shifted_series2)
        correlations[f"Lag_{lag}"] = corr
        
    return correlations

def calculate_correlation_matrix(
    data_dict: dict,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate a correlation matrix for a dictionary of TimeSeries.
    Args:
        data_dict: Dictionary where key is Ticker and value is Series (Close price).
    """
    df = pd.DataFrame(data_dict)
    return df.corr(method=method)

def detect_regime_changes(
    series1: pd.Series, 
    series2: pd.Series, 
    threshold: float = 0.5,
    window: int = 30
) -> pd.DataFrame:
    """
    Identify periods where correlation shifts drastically.
    Returns a DataFrame with Rolling Correlation and a 'Regime' column.
    Regime 1: High Correlation (> threshold)
    Regime 0: Low/Negative Correlation (<= threshold)
    """
    roll_corr = rolling_correlation(series1, series2, window=window)
    regime = roll_corr.apply(lambda x: 1 if x > threshold else 0)
    
    return pd.DataFrame({
        'Rolling_Corr': roll_corr,
        'Regime': regime
    })

def calculate_ccf(series1: pd.Series, series2: pd.Series, max_lags: int = 20) -> pd.Series:
    """
    Calculate Cross-Correlation Function for a range of lags.
    Positive lag means series1 leads series2.
    Negative lag means series2 leads series1.
    """
    lags = range(-max_lags, max_lags + 1)
    ccf_values = []
    
    for l in lags:
        if l < 0:
            # series2 leads series1: correlate s1 with s2 shifted back
            corr = series1.corr(series2.shift(abs(l)))
        elif l > 0:
            # series1 leads series2: correlate s1 shifted back with s2
            corr = series1.shift(l).corr(series2)
        else:
            corr = series1.corr(series2)
        ccf_values.append(corr)
        
    return pd.Series(ccf_values, index=lags)

def find_best_lag(series1: pd.Series, series2: pd.Series, max_lags: int = 20):
    """
    Find the lag with the highest absolute correlation.
    """
    ccf = calculate_ccf(series1, series2, max_lags)
    best_lag = ccf.abs().idxmax()
    return best_lag, ccf[best_lag]



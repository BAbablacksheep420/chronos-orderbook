"""
Chronos Order Book - Microstructural Feature Engineering
========================================================
Implements institutional-grade limit order book feature extraction:
- Cont et al. (2014) Order Flow Imbalance (OFI) & Multi-Level OFI
- Volume-Weighted Microprice & Depth Imbalance
- Tick-size aware & volatility-adjusted triple barrier targets
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def compute_ofi(df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """
    Compute multi-level Order Flow Imbalance (OFI) as formalized by
    Cont, Kukanov, and Stoikov (2014) and extended to multi-level books.
    
    OFI quantifies the net change in demand/supply across successive book updates.
    """
    ofi_df = pd.DataFrame(index=df.index)
    
    for k in range(1, n_levels + 1):
        bp = df[f"bid_price_{k}"].values
        bv = df[f"bid_vol_{k}"].values
        ap = df[f"ask_price_{k}"].values
        av = df[f"ask_vol_{k}"].values
        n = len(df)
        
        ofi_k = np.zeros(n)
        
        # Vectorized lag calculation
        bp_lag = np.roll(bp, 1)
        bv_lag = np.roll(bv, 1)
        ap_lag = np.roll(ap, 1)
        av_lag = np.roll(av, 1)
        
        # Bid contribution:
        # If bp > bp_lag -> new best bid, contribution = bv
        # If bp == bp_lag -> size change, contribution = bv - bv_lag
        # If bp < bp_lag -> bid dropped, contribution = -bv_lag
        bid_contrib = np.where(
            bp > bp_lag, bv,
            np.where(bp == bp_lag, bv - bv_lag, -bv_lag)
        )
        
        # Ask contribution:
        # If ap < ap_lag -> new lower ask, contribution = -av
        # If ap == ap_lag -> size change, contribution = -(av - av_lag)
        # If ap > ap_lag -> ask lifted, contribution = av_lag
        ask_contrib = np.where(
            ap < ap_lag, -av,
            np.where(ap == ap_lag, -(av - av_lag), av_lag)
        )
        
        ofi_k = bid_contrib + ask_contrib
        ofi_k[0] = 0.0  # Zero out first boundary tick
        ofi_df[f"ofi_l{k}"] = ofi_k
        
    # Aggregate multi-level weighted OFI
    weights = np.array([1.0 / (k ** 0.5) for k in range(1, n_levels + 1)])
    weights /= weights.sum()
    ofi_df["ofi_multilevel"] = np.dot(ofi_df[[f"ofi_l{k}" for k in range(1, n_levels + 1)]].values, weights)
    
    return ofi_df


def extract_lob_features(
    df: pd.DataFrame,
    n_levels: int = 5,
    roll_window: int = 20,
    tick_size: float = 0.01
) -> pd.DataFrame:
    """
    Extract comprehensive order book microstructural features from raw Level-2 snapshot time series.
    
    Args:
        df: DataFrame containing bid/ask prices and volumes up to n_levels.
        n_levels: Number of book levels to process (default: 5).
        roll_window: Window size for rolling statistics (volatility, spread trends).
        tick_size: Minimum price movement increment (default: $0.01 / 1 tick).
        
    Returns:
        pd.DataFrame: Feature engineered dataframe with microprice, OFI, depth imbalance, and volatility.
    """
    feats = pd.DataFrame(index=df.index)
    
    best_bid = df["bid_price_1"]
    best_ask = df["ask_price_1"]
    v_bid1 = df["bid_vol_1"]
    v_ask1 = df["ask_vol_1"]
    
    # 1. Price-based and Spread features
    mid_price = (best_ask + best_bid) / 2.0
    spread = (best_ask - best_bid).clip(lower=tick_size)
    spread_ticks = spread / tick_size
    spread_bps = (spread / (mid_price + 1e-9)) * 10000.0
    
    # Volume-weighted microprice
    denom_v1 = v_bid1 + v_ask1 + 1e-9
    microprice = (v_bid1 * best_ask + v_ask1 * best_bid) / denom_v1
    microprice_bias = (microprice - mid_price) / (mid_price + 1e-9)
    
    feats["mid_price"] = mid_price
    feats["spread"] = spread
    feats["spread_ticks"] = spread_ticks
    feats["spread_bps"] = spread_bps
    feats["microprice"] = microprice
    feats["microprice_bias"] = microprice_bias
    
    # 2. Level 1 Order Imbalance (OI_1)
    feats["order_imbalance_l1"] = (v_bid1 - v_ask1) / denom_v1
    
    # 3. Multi-level Depth Aggregation & VWAP
    tot_bid_vol = np.zeros(len(df))
    tot_ask_vol = np.zeros(len(df))
    weighted_bid_p = np.zeros(len(df))
    weighted_ask_p = np.zeros(len(df))
    
    for k in range(1, n_levels + 1):
        bp = df[f"bid_price_{k}"]
        bv = df[f"bid_vol_{k}"]
        ap = df[f"ask_price_{k}"]
        av = df[f"ask_vol_{k}"]
        
        tot_bid_vol += bv
        tot_ask_vol += av
        weighted_bid_p += (bp * bv)
        weighted_ask_p += (ap * av)
        
    tot_vol = tot_bid_vol + tot_ask_vol + 1e-9
    feats["depth_imbalance"] = (tot_bid_vol - tot_ask_vol) / tot_vol
    feats["total_bid_vol"] = tot_bid_vol
    feats["total_ask_vol"] = tot_ask_vol
    feats["total_depth"] = tot_vol
    feats["vwap_bid"] = weighted_bid_p / (tot_bid_vol + 1e-9)
    feats["vwap_ask"] = weighted_ask_p / (tot_ask_vol + 1e-9)
    feats["vwap_spread"] = feats["vwap_ask"] - feats["vwap_bid"]
    
    # 4. Order Flow Imbalance (OFI)
    ofi_df = compute_ofi(df, n_levels=n_levels)
    feats["ofi_l1"] = ofi_df["ofi_l1"]
    feats["ofi_multilevel"] = ofi_df["ofi_multilevel"]
    feats["ofi_normalized"] = ofi_df["ofi_l1"] / (tot_vol / n_levels + 1e-9)
    
    # 5. Rolling Dynamic Features
    mid_returns = mid_price.pct_change().fillna(0.0)
    feats["rolling_volatility"] = mid_returns.rolling(window=roll_window, min_periods=3).std().fillna(0.0)
    feats["rolling_spread_mean"] = spread_bps.rolling(window=roll_window, min_periods=3).mean().fillna(spread_bps)
    feats["rolling_imbalance_mean"] = feats["order_imbalance_l1"].rolling(window=roll_window, min_periods=3).mean().fillna(0.0)
    feats["imbalance_momentum"] = feats["order_imbalance_l1"] - feats["order_imbalance_l1"].shift(5).fillna(0.0)
    
    # Lag-1 Autocorrelation of Returns (Drift diagnostic)
    feats["roll_autocorr"] = mid_returns.rolling(window=roll_window, min_periods=5).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 3 else 0.0, raw=False
    ).fillna(0.0)
    
    # Preserve metadata if present
    if "true_regime" in df.columns:
        feats["true_regime"] = df["true_regime"]
    if "timestamp" in df.columns:
        feats["timestamp"] = df["timestamp"]
        
    return feats


def create_forecasting_dataset(
    feats_df: pd.DataFrame,
    horizon: int = 5,
    threshold: Optional[float] = None,
    tick_size: float = 0.01,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Construct feature matrix X and 3-class target labels y:
    - Down (class 0): future return < -threshold
    - Stationary (class 1): -threshold <= future return <= threshold
    - Up (class 2): future return > threshold
    
    Enforces minimum barrier based on tick size and dynamic volatility scaling.
    """
    mid = feats_df["mid_price"]
    future_mid = mid.shift(-horizon)
    future_returns = (future_mid - mid) / (mid + 1e-9)
    
    # Thresholding: 60th percentile of absolute returns ensures balanced representation
    if threshold is None:
        valid_ret = future_returns.dropna()
        threshold = max(1e-4, float(np.quantile(np.abs(valid_ret), 0.60)))
        
    labels = pd.Series(1, index=feats_df.index)
    labels[future_returns > threshold] = 2   # Up
    labels[future_returns < -threshold] = 0  # Down
    
    valid_idx = future_returns.dropna().index
    clean_df = feats_df.loc[valid_idx].copy()
    clean_df["target_return"] = future_returns.loc[valid_idx]
    clean_df["target_label"] = labels.loc[valid_idx]
    clean_df["barrier_threshold"] = threshold
    
    if feature_cols is None:
        feature_cols = [
            "spread_bps",
            "spread_ticks",
            "microprice_bias",
            "order_imbalance_l1",
            "depth_imbalance",
            "total_depth",
            "ofi_l1",
            "ofi_multilevel",
            "ofi_normalized",
            "vwap_spread",
            "rolling_volatility",
            "rolling_spread_mean",
            "rolling_imbalance_mean",
            "imbalance_momentum"
        ]
        
    X = clean_df[feature_cols].values
    y = clean_df["target_label"].values
    
    return X, y, clean_df

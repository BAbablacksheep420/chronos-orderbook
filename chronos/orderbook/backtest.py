"""
Chronos Order Book - Institutional Execution Simulator & Implementation Shortfall Backtester
=============================================================================================
Simulates mid-frequency (100ms - 5s) order book execution under realistic market frictions:
- Dynamic, book-driven bid-ask spread crossing from instantaneous LOB state
- Binance USDⓈ-M exchange fees (VIP0: 5.0 bps taker per side / 2.0 bps maker per side)
  Ref: https://www.binance.com/en/fee/trading — verified Oct 2026
  Note: VIP0 spot is 10 bps/side. Futures taker at VIP9 is ~1.7 bps.
- Implementation Shortfall (IS) vs Arrival Price for parent execution orders (Static TWAP vs Adaptive TWAP)
- Controlled ablation and adverse selection quantification per regime
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


def simulate_execution(
    eval_df: pd.DataFrame,
    taker_fee_bps: float = 5.0,
    horizon: int = 5,
    initial_capital: float = 100000.0
) -> Dict[str, Any]:
    """
    Directional execution simulator comparing Static Baseline vs Adaptive Conformal Selective Head.
    
    Fee assumptions (Binance USDⓈ-M Perpetual, VIP0 tier as of Oct 2026):
    - Taker: 5.0 bps per side → 10.0 bps round trip.
    - Maker: 2.0 bps per side → 4.0 bps round trip (not used here; taker-only model).
    - At VIP9: taker ~1.7 bps/side. At VIP0 spot: 10 bps/side.
    
    Friction per trade:
    - Entry: half-spread (from book) + taker fee.
    - Exit:  half-spread (from book) + taker fee.
    - Total: 2 * half_spread + 2 * taker_fee.
    
    WARNING: A 5-second BTC move has σ ≈ 2 bps against ≥10 bps taker round trip.
    A taker-directional strategy needs to predict ~5σ moves to break even.
    This simulator exists to quantify the gap, not to claim profitability.
    """
    n = len(eval_df)
    mid_prices = eval_df["mid_price"].values
    regimes = eval_df["true_regime"].values if "true_regime" in eval_df.columns else np.zeros(n)
    
    # Book-driven spread in basis points: instantaneous crossing cost
    if "spread_bps" in eval_df.columns:
        half_spread_bps = 0.5 * eval_df["spread_bps"].values
    elif "ask_price_1" in eval_df.columns and "bid_price_1" in eval_df.columns:
        half_spread_bps = 0.5 * ((eval_df["ask_price_1"].values - eval_df["bid_price_1"].values) / (mid_prices + 1e-9)) * 10000.0
    else:
        # Default conservative floor if book levels unavailable
        half_spread_bps = np.full(n, 1.5)
        
    base_preds = eval_df["baseline_pred"].values
    adapt_preds = eval_df["adaptive_pred"].values
    
    # 0 = Down (Short), 1 = Stationary (Cash), 2 = Up (Long)
    base_signals = np.where(base_preds == 2, 1, np.where(base_preds == 0, -1, 0))
    adapt_signals = np.where(adapt_preds == 2, 1, np.where(adapt_preds == 0, -1, 0))
    
    # Forward price return over horizon
    future_mid = np.roll(mid_prices, -horizon)
    raw_returns = (future_mid - mid_prices) / mid_prices
    raw_returns[-horizon:] = 0.0
    
    # Dynamic round-trip friction: entry half-spread + exit half-spread + 2 * taker fee
    round_trip_friction_bps = (2.0 * taker_fee_bps) + (2.0 * half_spread_bps)
    friction_decimal = round_trip_friction_bps / 10000.0
    
    base_gross = base_signals * raw_returns
    adapt_gross = adapt_signals * raw_returns
    
    base_net = np.where(base_signals != 0, base_gross - friction_decimal, 0.0)
    adapt_net = np.where(adapt_signals != 0, adapt_gross - friction_decimal, 0.0)
    
    base_equity = initial_capital * np.cumprod(1.0 + base_net)
    adapt_equity = initial_capital * np.cumprod(1.0 + adapt_net)
    
    def compute_stats(net_rets: np.ndarray) -> Dict[str, float]:
        active_mask = (net_rets != 0.0)
        n_trades = int(np.sum(active_mask))
        if n_trades == 0:
            return {"total_pnl_bps": 0.0, "win_rate": 0.0, "sharpe": 0.0, "max_dd_bps": 0.0, "trades": 0}
            
        trade_rets = net_rets[active_mask]
        total_pnl = float(np.sum(trade_rets) * 10000.0)
        win_rate = float(np.mean(trade_rets > 0)) * 100.0
        
        if n_trades < 2:
            sharpe = 0.0
        else:
            std_ret = float(np.std(trade_rets, ddof=1))
            if std_ret < 1e-7:
                sharpe = 0.0
            else:
                sharpe = float((np.mean(trade_rets) / std_ret) * np.sqrt(n_trades))
        
        cum = np.cumsum(net_rets)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = float(np.max(dd) * 10000.0)
        
        return {
            "total_pnl_bps": total_pnl,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "max_dd_bps": max_dd,
            "trades": n_trades
        }
        
    overall_base = compute_stats(base_net)
    overall_adapt = compute_stats(adapt_net)
    
    regime_pnl = {}
    for r in np.unique(regimes):
        mask = (regimes == r)
        regime_pnl[f"regime_{r}"] = {
            "baseline_pnl_bps": float(np.sum(base_net[mask]) * 10000.0),
            "adaptive_pnl_bps": float(np.sum(adapt_net[mask]) * 10000.0),
            "baseline_trades": int(np.sum(base_signals[mask] != 0)),
            "adaptive_trades": int(np.sum(adapt_signals[mask] != 0)),
            "capital_saved_bps": float(np.sum(adapt_net[mask] - base_net[mask]) * 10000.0)
        }
        
    backtest_df = eval_df.copy()
    backtest_df["base_net_pnl"] = base_net
    backtest_df["adapt_net_pnl"] = adapt_net
    backtest_df["base_equity"] = base_equity
    backtest_df["adapt_equity"] = adapt_equity
    
    return {
        "baseline": overall_base,
        "adaptive": overall_adapt,
        "pnl_advantage_bps": overall_adapt["total_pnl_bps"] - overall_base["total_pnl_bps"],
        "regime_breakdown": regime_pnl,
        "backtest_df": backtest_df
    }


def simulate_implementation_shortfall(
    eval_df: pd.DataFrame,
    total_quantity: float = 1000.0,
    taker_fee_bps: float = 5.0,
    side: str = "BUY"
) -> Dict[str, Any]:
    """
    Institutional Execution Benchmark: Implementation Shortfall (IS) vs Static TWAP.
    
    In institutional order execution, algorithms cannot permanently abstain from trading.
    They must execute the parent order quantity Q over time horizon T.
    
    - Static TWAP: Dispatches constant slice q_t = Q / T at every tick, regardless of regime.
    - Drift-Aware Adaptive TWAP: Dynamically scales down participation during detected liquidity
      shocks / widened spreads (e.g. 20% slice rate) and catches up when spreads normalize.
      
    Both strategies pay identical exchange fees on the completed parent order.
    The evaluation metric is Implementation Shortfall relative to Arrival Price P_0:
    IS (bps) = ((Avg_Execution_Price - P_0) / P_0) * 10,000 for BUY.
    """
    n = len(eval_df)
    if n < 10:
        return {}
        
    mid_prices = eval_df["mid_price"].values
    p_arrival = mid_prices[0]
    
    if "spread_bps" in eval_df.columns:
        half_spread_frac = 0.5 * (eval_df["spread_bps"].values / 10000.0)
    else:
        half_spread_frac = np.full(n, 0.00015)
        
    # Execution price per tick: buyer crosses ask = mid * (1 + half_spread)
    if side.upper() == "BUY":
        exec_prices = mid_prices * (1.0 + half_spread_frac)
    else:
        exec_prices = mid_prices * (1.0 - half_spread_frac)
        
    # 1. Static TWAP: uniform allocation
    static_slices = np.full(n, total_quantity / n)
    static_spent = np.sum(static_slices * exec_prices)
    avg_price_static = static_spent / total_quantity
    
    # 2. Adaptive TWAP: scale down participation during detected stress
    c_mults = eval_df["conviction"].values if "conviction" in eval_df.columns else np.ones(n)
    shift_flags = eval_df["shift_flag"].values if "shift_flag" in eval_df.columns else np.zeros(n)
    
    # Participation weights: throttled down to 20% when shift_flag is True or conviction low
    raw_weights = np.where(shift_flags, 0.20, np.clip(c_mults, 0.40, 1.0))
    adaptive_weights = raw_weights / np.sum(raw_weights)
    adaptive_slices = adaptive_weights * total_quantity
    
    adaptive_spent = np.sum(adaptive_slices * exec_prices)
    avg_price_adaptive = adaptive_spent / total_quantity
    
    # Implementation Shortfall in basis points
    sign = 1.0 if side.upper() == "BUY" else -1.0
    is_static_bps = sign * ((avg_price_static - p_arrival) / p_arrival) * 10000.0
    is_adaptive_bps = sign * ((avg_price_adaptive - p_arrival) / p_arrival) * 10000.0
    shortfall_saved_bps = is_static_bps - is_adaptive_bps
    
    fee_bps = taker_fee_bps
    
    return {
        "arrival_price": float(p_arrival),
        "static_twap": {
            "avg_price": float(avg_price_static),
            "implementation_shortfall_bps": float(is_static_bps),
            "total_fee_bps": float(fee_bps)
        },
        "adaptive_twap": {
            "avg_price": float(avg_price_adaptive),
            "implementation_shortfall_bps": float(is_adaptive_bps),
            "total_fee_bps": float(fee_bps)
        },
        "shortfall_reduction_bps": float(shortfall_saved_bps),
        "fill_rate": 100.0
    }


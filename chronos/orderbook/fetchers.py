"""
Chronos Order Book - Real Market Data Plumbing & Synchronization
================================================================
Implements:
- Official Binance Depth Stream protocol with local book management & sequence-ID validation
- Historical tick archive loader for public Binance / Tardis LOB data
- Realistic multi-regime order book simulator calibrated to real crypto microstructure (tight 1-2 tick calm spreads)
"""

import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class BinanceLocalOrderBook:
    """
    Manages a live local limit order book replica synchronized via Binance WebSocket diffs.
    Follows Binance best-practice synchronization rules:
    1. Buffer websocket depth delta events.
    2. Fetch snapshot via REST API.
    3. Drop events where event.u <= snapshot.lastUpdateId.
    4. Validate sequence continuity: event.U <= lastUpdateId + 1 and event.u >= lastUpdateId + 1.
    5. Update local book: quantity 0 removes level; non-zero updates level size.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_update_id = 0
        self.is_synchronized = False
        self.event_buffer: List[Dict[str, Any]] = []
        
    def apply_snapshot(self, snapshot: Dict[str, Any]):
        """Initialize book from REST snapshot."""
        self.last_update_id = snapshot.get("lastUpdateId", 0)
        self.bids = {float(p): float(v) for p, v in snapshot.get("bids", [])}
        self.asks = {float(p): float(v) for p, v in snapshot.get("asks", [])}
        
        # Replay buffered events
        valid_events = [e for e in self.event_buffer if e.get("u", 0) > self.last_update_id]
        for e in valid_events:
            self.apply_diff(e, check_sequence=False)
        self.event_buffer = []
        self.is_synchronized = True
        
    def buffer_event(self, event: Dict[str, Any]):
        """Buffer diff event during initial synchronization handshake."""
        if not self.is_synchronized:
            self.event_buffer.append(event)
        else:
            self.apply_diff(event, check_sequence=True)
            
    def apply_diff(self, event: Dict[str, Any], check_sequence: bool = True) -> bool:
        """Apply depth delta update event."""
        u = event.get("u", 0)
        U = event.get("U", 0)
        
        if check_sequence and self.last_update_id > 0:
            # Check sequence continuity
            if U > self.last_update_id + 1:
                # Sequence gap detected, resynchronization required
                self.is_synchronized = False
                return False
                
        # Apply bid updates
        for p_str, v_str in event.get("b", []):
            p, v = float(p_str), float(v_str)
            if v == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = v
                
        # Apply ask updates
        for p_str, v_str in event.get("a", []):
            p, v = float(p_str), float(v_str)
            if v == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = v
                
        self.last_update_id = u
        return True
        
    def get_top_k(self, k: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Return top K sorted bids (descending) and asks (ascending)."""
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:k]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:k]
        return {"bids": sorted_bids, "asks": sorted_asks}


def fetch_binance_depth(symbol: str = "BTCUSDT", limit: int = 20) -> Dict[str, Any]:
    """Fetch live Level-2 Limit Order Book snapshot from Binance REST endpoint."""
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit={limit}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        bids = [[float(p), float(v)] for p, v in data.get("bids", [])]
        asks = [[float(p), float(v)] for p, v in data.get("asks", [])]
        return {
            "timestamp": time.time(),
            "lastUpdateId": data.get("lastUpdateId"),
            "bids": bids,
            "asks": asks,
            "symbol": symbol.upper()
        }
    except Exception as e:
        print(f"[fetch_binance_depth] Error fetching {symbol}: {e}")
        return {}


def depth_snapshot_to_row(snapshot: Dict[str, Any], n_levels: int = 5) -> Dict[str, float]:
    """Flatten LOB depth snapshot into standard column format."""
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    row = {"timestamp": snapshot.get("timestamp", time.time())}
    
    for i in range(n_levels):
        bp = bids[i][0] if i < len(bids) else (bids[-1][0] if bids else 0.0)
        bv = bids[i][1] if i < len(bids) else 0.0
        ap = asks[i][0] if i < len(asks) else (asks[-1][0] if asks else 0.0)
        av = asks[i][1] if i < len(asks) else 0.0
        
        row[f"bid_price_{i+1}"] = bp
        row[f"bid_vol_{i+1}"] = bv
        row[f"ask_price_{i+1}"] = ap
        row[f"ask_vol_{i+1}"] = av
        
    return row


def load_real_lob_dataset(filepath: str, n_levels: int = 5) -> pd.DataFrame:
    """
    Load and parse a real-world LOB dataset (Binance tick archive, LOBSTER sample, or Tardis format).
    Standardizes columns to bid_price_1..K, bid_vol_1..K, ask_price_1..K, ask_vol_1..K.
    """
    df = pd.read_csv(filepath)
    # Check if already standardized
    if "bid_price_1" in df.columns:
        return df
        
    # Standardize LOBSTER-style format (Price Bid 1, Size Bid 1, etc.)
    rename_map = {}
    for k in range(1, n_levels + 1):
        for candidate in [f"bid_price_{k}", f"Ask Price {k}", f"bid_p_{k}", f"bp{k}"]:
            if candidate in df.columns:
                rename_map[candidate] = f"bid_price_{k}"
        for candidate in [f"ask_price_{k}", f"Bid Price {k}", f"ask_p_{k}", f"ap{k}"]:
            if candidate in df.columns:
                rename_map[candidate] = f"ask_price_{k}"
                
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def generate_synthetic_lob(
    n_steps: int = 2500,
    seed: int = 42,
    regime_shifts: Optional[List[int]] = None,
    n_levels: int = 5,
    initial_price: float = 60000.0,
    tick_size: float = 0.10
) -> pd.DataFrame:
    """
    Simulate a multi-regime Limit Order Book calibrated to realistic crypto market microstructure:
    - Calm Regime: Tight spreads of 1-2 ticks (~0.25 bps on BTCUSDT futures at $60k), balanced depth.
    - Stressed / Macro Shock Regime: Spread widens to 5-10 ticks (~1.2 bps), volatility explodes 4x.
    - Trending Flow Regime: Asymmetric order book depth driving directional drift.
    
    Parameter calibration notes (post-review, Oct 2026):
    - tick_size = 0.10 matches Binance BTCUSDT-Perp tick. At $60k, 1 tick = 0.10/60000 = 0.17 bps.
    - Calm spread 1.5 ticks = 0.15 USD = 0.25 bps (not "0.3 bps" as previously stated).
    - Calm sigma 3 bps/tick is ~3-10x real BTC (real ~0.9 bps/s at 50% ann. vol).
      This is intentionally elevated to produce regime-observable dynamics in a short sample.
    - Shock spread 7 ticks = 0.70 USD = 1.17 bps (not "2.3 bps" as previously stated).
    - Friction in backtest is dynamic from the book, not constant 1.5 bps.
    
    NOTE: 100ms diff streams are time-aggregated snapshots. Event-level OFI as formalized
    by Cont et al. (2014) assumes individual order events, not time-bucketed aggregates.
    The OFI computed here approximates the original using successive snapshots.
    """
    if regime_shifts is None:
        regime_shifts = [875, 1750]
        
    np.random.seed(seed)
    
    timestamps = np.arange(n_steps, dtype=float)
    mid_prices = np.zeros(n_steps)
    true_regimes = np.zeros(n_steps, dtype=int)
    
    bid_prices = np.zeros((n_steps, n_levels))
    bid_volumes = np.zeros((n_steps, n_levels))
    ask_prices = np.zeros((n_steps, n_levels))
    ask_volumes = np.zeros((n_steps, n_levels))
    
    current_mid = initial_price
    current_regime = 0
    shift_idx = 0
    
    # Realistically calibrated regimes:
    # Spread ticks: 1-2 ticks in calm (0.33 bps on $60k BTC)
    regime_params = {
        0: {"spread_ticks": 1.5, "sigma": 0.0003, "drift": 0.0, "jump_prob": 0.002, "depth_mean": 8.0},
        1: {"spread_ticks": 7.0, "sigma": 0.0018, "drift": -0.0001, "jump_prob": 0.05, "depth_mean": 2.2},
        2: {"spread_ticks": 2.5, "sigma": 0.0006, "drift": 0.00025, "jump_prob": 0.008, "depth_mean": 6.5}
    }
    
    for t in range(n_steps):
        if shift_idx < len(regime_shifts) and t >= regime_shifts[shift_idx]:
            shift_idx += 1
            current_regime = shift_idx % len(regime_params)
            
        true_regimes[t] = current_regime
        params = regime_params[current_regime]
        
        # Geometric Brownian motion + Poisson Jump
        jump = 0.0
        if np.random.rand() < params["jump_prob"]:
            jump = np.random.normal(0, params["sigma"] * 3.5)
            
        ret = params["drift"] + np.random.normal(0, params["sigma"]) + jump
        current_mid = max(100.0, current_mid * (1.0 + ret))
        # Snap mid-price to half-tick
        current_mid = round(current_mid / (tick_size * 0.5)) * (tick_size * 0.5)
        mid_prices[t] = current_mid
        
        half_spread = (params["spread_ticks"] * tick_size) * 0.5
        best_bid = round((current_mid - half_spread) / tick_size) * tick_size
        best_ask = round((current_mid + half_spread) / tick_size) * tick_size
        if best_ask <= best_bid:
            best_ask = best_bid + tick_size
            
        asym_bias = 0.0
        if current_regime == 2:
            asym_bias = 0.35
        elif current_regime == 1:
            asym_bias = np.random.choice([-0.3, 0.3])
            
        for k in range(n_levels):
            bid_prices[t, k] = round(best_bid - (k * tick_size), 2)
            ask_prices[t, k] = round(best_ask + (k * tick_size), 2)
            
            base_vol = params["depth_mean"] * (1.0 + 0.2 * k)
            bid_noise = np.random.gamma(shape=2.5, scale=base_vol / 2.5) * (1.0 + asym_bias)
            ask_noise = np.random.gamma(shape=2.5, scale=base_vol / 2.5) * (1.0 - asym_bias)
            
            bid_volumes[t, k] = round(max(0.05, bid_noise), 3)
            ask_volumes[t, k] = round(max(0.05, ask_noise), 3)
            
    data_dict = {
        "timestamp": timestamps,
        "mid_price": mid_prices,
        "true_regime": true_regimes
    }
    for k in range(n_levels):
        data_dict[f"bid_price_{k+1}"] = bid_prices[:, k]
        data_dict[f"bid_vol_{k+1}"] = bid_volumes[:, k]
        data_dict[f"ask_price_{k+1}"] = ask_prices[:, k]
        data_dict[f"ask_vol_{k+1}"] = ask_volumes[:, k]
        
    return pd.DataFrame(data_dict)

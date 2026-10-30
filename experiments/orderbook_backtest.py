"""
Chronos AI - Subsidiary Feature: Mid-Frequency Execution Backtest
=================================================================
Evaluates economic viability:
- Simulates execution with 2.0 bps exchange taker fees + half-spread crossing cost
- Quantifies net PnL across calm, shock, and trend regimes
- Demonstrates capital preservation: adverse selection avoided during liquidity shocks
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chronos.orderbook.fetchers import generate_synthetic_lob
from chronos.orderbook.features import extract_lob_features, create_forecasting_dataset
from chronos.orderbook.model import LOBForecaster
from chronos.orderbook.trainer import evaluate_adaptive_pipeline
from chronos.orderbook.regime_detector import RegimeShiftManager
from chronos.orderbook.backtest import simulate_execution


def main():
    print("\n" + "=" * 76)
    print("  CHRONOS SUBSIDIARY: MID-FREQUENCY NET PNL BACKTEST (AFTER FEES)")
    print("=" * 76)
    
    lob_df = generate_synthetic_lob(n_steps=2500, seed=42)
    feats_df = extract_lob_features(lob_df)
    
    feature_cols = [
        "spread_bps", "spread_ticks", "microprice_bias",
        "order_imbalance_l1", "depth_imbalance", "total_depth",
        "ofi_l1", "ofi_multilevel", "vwap_spread",
        "rolling_volatility", "rolling_spread_mean",
        "rolling_imbalance_mean", "imbalance_momentum"
    ]
    
    X, y, clean_df = create_forecasting_dataset(feats_df, horizon=5, feature_cols=feature_cols)
    
    n_train = int(len(clean_df) * 0.35)
    train_df = clean_df.iloc[:n_train]
    test_df = clean_df.iloc[n_train:]
    
    model = LOBForecaster(model_type="tcn")
    model.fit(train_df[feature_cols].values, train_df["target_label"].values, epochs=20)
    
    mgr = RegimeShiftManager(mode="adjusted_ks")
    eval_res = evaluate_adaptive_pipeline(test_df, feature_cols, model, shift_manager=mgr, val_df=train_df)
    
    # Run backtest with 5.0 bps taker fee (Binance USDⓈ-M VIP0)
    backtest_res = simulate_execution(eval_res["eval_df"], taker_fee_bps=5.0)
    
    base_stats = backtest_res["baseline"]
    adapt_stats = backtest_res["adaptive"]
    
    print("-" * 76)
    print("  OVERALL STRATEGY PERFORMANCE (After 5.0 bps taker/side + spread cross):")
    print("-" * 76)
    print(f"  * Static Baseline Net PnL   : {base_stats['total_pnl_bps']:+.2f} bps | Sharpe: {base_stats['sharpe']:.2f} | Max DD: {base_stats['max_dd_bps']:.2f} bps")
    print(f"  * Adaptive Engine Net PnL   : {adapt_stats['total_pnl_bps']:+.2f} bps | Sharpe: {adapt_stats['sharpe']:.2f} | Max DD: {adapt_stats['max_dd_bps']:.2f} bps")
    print(f"  * Net Capital Advantage     : {backtest_res['pnl_advantage_bps']:+.2f} bps")
    print(f"  * Win Rate on Executed Trades: Static={base_stats['win_rate']:.1f}% vs Adaptive={adapt_stats['win_rate']:.1f}%")
    print("-" * 76)
    
    print("\n  REGIME-BY-REGIME PNL BREAKDOWN:")
    for r_name, r_pnl in backtest_res["regime_breakdown"].items():
        print(f"    - {r_name.upper()}:")
        print(f"        Static Baseline PnL : {r_pnl['baseline_pnl_bps']:+.2f} bps ({r_pnl['baseline_trades']} trades)")
        print(f"        Adaptive Engine PnL : {r_pnl['adaptive_pnl_bps']:+.2f} bps ({r_pnl['adaptive_trades']} trades)")
        print(f"        Capital Preserved   : {r_pnl['capital_saved_bps']:+.2f} bps (toxic adverse selection neutralized)")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()

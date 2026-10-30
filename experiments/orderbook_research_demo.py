"""
Chronos AI - Subsidiary Feature: Order Book Forecasting Under Regime Shifts Demo
================================================================================
This script runs the complete end-to-end research workflow for the LOB-RS subsidiary engine:
1. Simulates multi-regime high-frequency Limit Order Book snapshots (Calm, Macro Shock, Trending Flow).
2. Extracts microstructural features (Order Imbalance, Microprice, Depth Imbalance, Realized Volatility).
3. Trains baseline model (TCN / Neural Network) on initial calm regime.
4. Performs streaming chronological evaluation using online Kolmogorov-Smirnov & Page-Hinkley shift detection.
5. Dynamically adapts predictions and discounts overconfidence during regime transitions.
6. Evaluates comparative performance metrics (Accuracy, F1, Latency).
7. Renders the interactive 4-panel diagnostic Plotly dashboard.
"""

import sys
import os
import numpy as np

# Ensure chronos package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chronos.orderbook.fetchers import generate_synthetic_lob
from chronos.orderbook.features import extract_lob_features, create_forecasting_dataset
from chronos.orderbook.trainer import train_lob_model, evaluate_adaptive_pipeline
from chronos.orderbook.regime_detector import RegimeShiftManager
from chronos.orderbook.visualizer import plot_lob_regime_analysis


def run_demo(n_steps: int = 2500, model_type: str = "tcn", output_html: str = "orderbook_regime_analysis.html"):
    print("\n" + "=" * 72)
    print("  CHRONOS AI SUBSIDIARY: ORDER BOOK FORECASTING UNDER REGIME SHIFTS")
    print("=" * 72)
    
    # 1. Simulate Order Book
    print(f"\n[Step 1/5] Simulating {n_steps} LOB steps with 3 distinct market regimes...")
    print("           - Regime 0 (0 -> 875): Calm, tight spread (2.5 bps), low vol, balanced depth.")
    print("           - Regime 1 (875 -> 1750): Macro Shock / FOMC, spread explosion (14 bps), jump diffusion.")
    print("           - Regime 2 (1750 -> 2500): Trending Flow, persistent queue asymmetry, momentum drift.")
    
    shifts = [int(n_steps * 0.35), int(n_steps * 0.70)]
    lob_df = generate_synthetic_lob(n_steps=n_steps, seed=42, regime_shifts=shifts)
    print(f"           Simulated {len(lob_df)} book snapshots with 5 bid and 5 ask levels.")
    
    # 2. Feature Extraction
    print("\n[Step 2/5] Extracting microstructural LOB features...")
    feats_df = extract_lob_features(lob_df)
    
    feature_cols = [
        "spread_bps", "microprice_bias", "order_imbalance_l1",
        "depth_imbalance", "total_bid_vol", "total_ask_vol",
        "vwap_spread", "rolling_volatility", "rolling_spread_mean",
        "rolling_imbalance_mean", "imbalance_momentum"
    ]
    
    X, y, clean_df = create_forecasting_dataset(feats_df, horizon=5, feature_cols=feature_cols)
    print(f"           Constructed feature matrix X: {X.shape}, Targets: Down={np.sum(y==0)}, Stat={np.sum(y==1)}, Up={np.sum(y==2)}")
    
    # 3. Model Training
    print(f"\n[Step 3/5] Fitting base {model_type.upper()} on initial stationary regime (first 35% of data)...")
    base_model, (X_train, y_train), (X_test, y_test) = train_lob_model(
        X, y, train_ratio=0.35, model_type=model_type, epochs=25
    )
    print("           Base forecaster successfully fitted.")
    
    # 4. Adaptive Evaluation
    print("\n[Step 4/5] Executing streaming chronological evaluation with online shift detection...")
    detector = RegimeShiftManager(ks_ref_window=250, ks_test_window=50, ks_alpha=0.01)
    results = evaluate_adaptive_pipeline(clean_df, feature_cols, base_model, detector)
    
    overall = results["overall"]
    print("\n" + "-" * 72)
    print("  EMPIRICAL RESEARCH EVALUATION RESULTS")
    print("-" * 72)
    print(f"  * Static Baseline Accuracy : {overall['baseline_accuracy']:.2%}")
    print(f"  * Adaptive Engine Accuracy : {overall['adaptive_accuracy']:.2%}  (Improvement: +{overall['accuracy_gain_pct']:.2f}%)")
    print(f"  * Static Baseline F1-Score : {overall['baseline_f1']:.4f}")
    print(f"  * Adaptive Engine F1-Score : {overall['adaptive_f1']:.4f}")
    print(f"  * Mean Inference Latency   : {overall['mean_latency_ms']:.3f} ms (p95: {overall['p95_latency_ms']:.3f} ms)")
    print("-" * 72)
    
    print("\n  Performance Breakdown By Regime:")
    for r_name, r_stats in results["by_regime"].items():
        print(f"    - {r_name.upper()} ({r_stats['count']} samples): Baseline Acc = {r_stats['baseline_acc']:.2%} | Adaptive Acc = {r_stats['adaptive_acc']:.2%}")
        
    # 5. Visualizer
    print(f"\n[Step 5/5] Exporting diagnostic Plotly dashboard to {output_html}...")
    plot_lob_regime_analysis(results["eval_df"], output_html=output_html)
    print("=" * 72)
    print(f"Demo complete! Interactive report available at: {output_html}\n")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "orderbook_regime_analysis.html"
    run_demo(output_html=out_file)

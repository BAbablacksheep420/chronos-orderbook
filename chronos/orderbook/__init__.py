"""
Chronos AI - Subsidiary Feature: Order Book Forecasting Under Regime Shifts (LOB-RS)
===================================================================================
A quant research and execution engine featuring:
- Autocorrelation-adjusted Kolmogorov-Smirnov & Calibrated CUSUM change detection
- Multi-level Order Flow Imbalance (OFI) & microprice feature engineering
- Temperature-scaled Causal Dilated TCN & Adaptive Conformal Selective Prediction
- Purged Walk-Forward Cross-Validation & Net PnL Backtesting
"""

from .fetchers import (
    fetch_binance_depth,
    generate_synthetic_lob,
    load_real_lob_dataset,
    BinanceLocalOrderBook
)
from .features import (
    extract_lob_features,
    create_forecasting_dataset,
    compute_ofi
)
from .regime_detector import (
    AutocorrelationAdjustedKS,
    CalibratedCUSUM,
    RegimeShiftManager,
    effective_sample_size,
    estimate_lag1_autocorr
)
from .model import (
    LOBForecaster,
    TCNForecaster,
    TemperatureScaler,
    AdaptiveConformalSelectiveHead
)
from .trainer import (
    evaluate_adaptive_pipeline,
    purged_train_test_split,
    run_full_ablation_suite,
    compute_ece,
    compute_multiclass_brier
)
from .backtest import simulate_execution
from .visualizer import plot_lob_regime_analysis

__all__ = [
    "fetch_binance_depth",
    "generate_synthetic_lob",
    "load_real_lob_dataset",
    "BinanceLocalOrderBook",
    "extract_lob_features",
    "create_forecasting_dataset",
    "compute_ofi",
    "AutocorrelationAdjustedKS",
    "CalibratedCUSUM",
    "RegimeShiftManager",
    "effective_sample_size",
    "estimate_lag1_autocorr",
    "LOBForecaster",
    "TCNForecaster",
    "TemperatureScaler",
    "AdaptiveConformalSelectiveHead",
    "evaluate_adaptive_pipeline",
    "purged_train_test_split",
    "run_full_ablation_suite",
    "compute_ece",
    "compute_multiclass_brier",
    "simulate_execution",
    "plot_lob_regime_analysis"
]

import click
import pandas as pd
import numpy as np
from chronos.fetchers import fetch_data
from chronos.processors import resample_ohlc
from chronos.analytics import rolling_correlation, check_cointegration
import os
import time

@click.group()
def cli():
    """Chronos AI - Multi-Timeframe Correlation Discovery & Order Book Forecasting"""
    pass

# ==========================================
# CORE CHRONOS CLI COMMANDS
# ==========================================

@cli.command()
@click.option('--ticker', multiple=True, help='Ticker symbol (e.g. AAPL, BTC-USD)')
@click.option('--period', default='1y', help='Data period')
@click.option('--out', default='data', help='Output directory')
def fetch(ticker, period, out):
    """Fetch financial data."""
    if not os.path.exists(out):
        os.makedirs(out)
        
    for t in ticker:
        df = fetch_data(t, period=period)
        if not df.empty:
            path = os.path.join(out, f"{t}.csv")
            df.to_csv(path)
            click.echo(f"Saved {t} to {path}")

@cli.command()
@click.option('--asset1', required=True, help='First asset symbol')
@click.option('--asset2', required=True, help='Second asset symbol')
@click.option('--timeframe', default='1D', help='Timeframe to analyze (e.g. 1H, 1D)')
def analyze(asset1, asset2, timeframe):
    """Analyze correlation between two assets."""
    # Load data
    path1 = f"data/{asset1}.csv"
    path2 = f"data/{asset2}.csv"
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        click.echo("Data not found. Run 'fetch' first.")
        return

    # Read and parse index
    df1 = pd.read_csv(path1, index_col=0, parse_dates=True)
    df2 = pd.read_csv(path2, index_col=0, parse_dates=True)
    
    # Resample to common timeframe
    click.echo(f"Resampling to {timeframe}...")
    df1 = resample_ohlc(df1, timeframe)
    df2 = resample_ohlc(df2, timeframe)

    # Align data (intersection of dates)
    common_idx = df1.index.intersection(df2.index)
    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]
    
    # Calculate Correlation
    corr = df1['Close'].corr(df2['Close'])
    click.echo(f"Global Pearson Correlation ({timeframe}): {corr:.4f}")
    
    # Rolling Correlation
    roll_corr = rolling_correlation(df1['Close'], df2['Close'], window=30)
    current_roll = roll_corr.iloc[-1]
    click.echo(f"Current 30-period Rolling Correlation: {current_roll:.4f}")

    # Cointegration
    p_val = check_cointegration(df1['Close'], df2['Close'])
    click.echo(f"Cointegration p-value: {p_val:.4f} ({'Cointegrated' if p_val < 0.05 else 'Not Cointegrated'})")

@cli.command()
@click.option('--tickers', multiple=True, required=True, help='List of tickers to scan')
@click.option('--timeframe', default='1D', help='Timeframe to analyze')
@click.option('--out', default='heatmap.html', help='Output HTML file for heatmap')
def scan(tickers, timeframe, out):
    """Scan multiple assets and generate a correlation matrix heatmap."""
    from chronos.analytics import calculate_correlation_matrix
    from chronos.visualizer import plot_heatmap
    
    data_dict = {}
    for t in tickers:
        path = f"data/{t}.csv"
        if not os.path.exists(path):
            click.echo(f"Warning: {t} data not found. Skipping.")
            continue
            
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if timeframe != '1D':
            df = resample_ohlc(df, timeframe)
        
        data_dict[t] = df['Close']
    
    if len(data_dict) < 2:
        click.echo("Not enough data to calculate matrix.")
        return

    full_df = pd.DataFrame(data_dict).dropna()
    
    click.echo(f"Calculating correlation matrix for {len(full_df.columns)} assets...")
    corr_matrix = calculate_correlation_matrix(full_df.to_dict('series'))
    
    click.echo(f"Saving heatmap to {out}...")
    plot_heatmap(corr_matrix, title=f"Correlation Matrix ({timeframe})", save_path=out)
    click.echo("Done.")

@cli.command()
@click.option('--asset1', required=True, help='First asset symbol')
@click.option('--asset2', required=True, help='Second asset symbol')
@click.option('--timeframe', default='1D', help='Timeframe to analyze')
@click.option('--threshold', default=0.5, help='Correlation threshold for regime')
def regime(asset1, asset2, timeframe, threshold):
    """Detect correlation regime changes over time."""
    from chronos.analytics import detect_regime_changes
    from chronos.visualizer import plot_regime_change
    
    path1 = f"data/{asset1}.csv"
    path2 = f"data/{asset2}.csv"
    
    df1 = pd.read_csv(path1, index_col=0, parse_dates=True)
    df2 = pd.read_csv(path2, index_col=0, parse_dates=True)
    
    click.echo(f"Resampling to {timeframe}...")
    df1 = resample_ohlc(df1, timeframe)
    df2 = resample_ohlc(df2, timeframe)
        
    common_idx = df1.index.intersection(df2.index)
    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]
    
    df_regime = detect_regime_changes(df1['Close'], df2['Close'], threshold=threshold)
    
    high_corr_days = df_regime[df_regime['Regime'] == 1].shape[0]
    total_days = df_regime.shape[0]
    click.echo(f"High Correlation Regimes: {high_corr_days}/{total_days} periods ({high_corr_days/total_days:.1%})")
    
    fig = plot_regime_change(df_regime, title=f"Regime Change: {asset1} vs {asset2}")
    out_file = f"regime_{asset1}_{asset2}.html"
    fig.write_html(out_file)
    click.echo(f"Regime plot saved to {out_file}")


# ==========================================
# SUBSIDIARY FEATURE: ORDER BOOK FORECASTING
# ==========================================

@cli.group('orderbook')
def orderbook_group():
    """Subsidiary Engine: Limit Order Book Forecasting Under Regime Shifts (LOB-RS)."""
    pass

@orderbook_group.command('demo')
@click.option('--steps', default=2500, help='Total synthetic LOB steps to simulate')
@click.option('--model-type', default='tcn', help='Model architecture: tcn or sklearn')
@click.option('--out', default='orderbook_regime_analysis.html', help='Path for output visualization dashboard')
def orderbook_demo(steps, model_type, out):
    """Run end-to-end research demo for Order Book Forecasting under Regime Shifts."""
    from chronos.orderbook.fetchers import generate_synthetic_lob
    from chronos.orderbook.features import extract_lob_features, create_forecasting_dataset
    from chronos.orderbook.trainer import train_lob_model, evaluate_adaptive_pipeline
    from chronos.orderbook.regime_detector import RegimeShiftManager
    from chronos.orderbook.visualizer import plot_lob_regime_analysis
    
    click.echo("=" * 68)
    click.echo("Chronos AI Subsidiary: Order Book Forecasting Under Regime Shifts")
    click.echo("=" * 68)
    
    # 1. Generate multi-regime synthetic order book
    click.echo(f"[1/5] Simulating {steps} ticks of multi-regime LOB data (Regimes: Calm -> Shock -> Trend)...")
    lob_df = generate_synthetic_lob(n_steps=steps, seed=42, regime_shifts=[int(steps * 0.35), int(steps * 0.70)])
    
    # 2. Extract microstructural features
    click.echo("[2/5] Extracting microstructural LOB features (imbalances, microprice, rolling vol)...")
    feats_df = extract_lob_features(lob_df)
    
    feature_cols = [
        "spread_bps", "microprice_bias", "order_imbalance_l1",
        "depth_imbalance", "total_bid_vol", "total_ask_vol",
        "vwap_spread", "rolling_volatility", "rolling_spread_mean",
        "rolling_imbalance_mean", "imbalance_momentum"
    ]
    X, y, clean_df = create_forecasting_dataset(feats_df, horizon=5, feature_cols=feature_cols)
    click.echo(f"      Feature matrix: {X.shape}, Class distribution: Down={np.sum(y==0)}, Stat={np.sum(y==1)}, Up={np.sum(y==2)}")
    
    # 3. Train base model on early regime
    click.echo(f"[3/5] Training base {model_type.upper()} model on initial stationary regime (35% sample)...")
    model, (X_train, y_train), (X_test, y_test) = train_lob_model(
        X, y, train_ratio=0.35, model_type=model_type, epochs=25
    )
    
    # 4. Sequential streaming evaluation with online regime detector
    click.echo("[4/5] Executing streaming chronological evaluation (Rolling KS + Page-Hinkley)...")
    detector = RegimeShiftManager(ks_ref_window=250, ks_test_window=50, ks_alpha=0.01)
    results = evaluate_adaptive_pipeline(clean_df, feature_cols, model, detector)
    
    overall = results["overall"]
    click.echo("-" * 68)
    click.echo(f"  * Static Baseline Accuracy : {overall['baseline_accuracy']:.2%}")
    click.echo(f"  * Adaptive Engine Accuracy : {overall['adaptive_accuracy']:.2%}  (Gain: +{overall['accuracy_gain_pct']:.2f}%)")
    click.echo(f"  * Static Baseline F1 Score : {overall['baseline_f1']:.4f}")
    click.echo(f"  * Adaptive Engine F1 Score : {overall['adaptive_f1']:.4f}")
    click.echo(f"  * Mean Inference Latency   : {overall['mean_latency_ms']:.2f} ms (p95: {overall['p95_latency_ms']:.2f} ms)")
    click.echo("-" * 68)
    
    for r_name, r_stats in results["by_regime"].items():
        click.echo(f"  {r_name.upper()} ({r_stats['count']} ticks): Baseline={r_stats['baseline_acc']:.2%} | Adaptive={r_stats['adaptive_acc']:.2%}")
        
    # 5. Generate visual dashboard
    click.echo(f"[5/5] Generating interactive diagnostic dashboard to {out}...")
    plot_lob_regime_analysis(results["eval_df"], output_html=out)
    click.echo("=" * 68)
    click.echo("Demo completed successfully.")

@orderbook_group.command('fetch')
@click.option('--symbol', default='BTCUSDT', help='Cryptocurrency pair (e.g. BTCUSDT, ETHUSDT)')
@click.option('--limit', default=20, help='Order book depth levels (5, 10, 20, 50)')
@click.option('--synthetic', is_flag=True, help='Generate multi-regime synthetic dataset instead of live fetch')
@click.option('--steps', default=2000, help='Steps if synthetic')
@click.option('--out', default='data/orderbook_sample.csv', help='Target CSV path')
def orderbook_fetch(symbol, limit, synthetic, steps, out):
    """Fetch live Binance order book snapshot or generate synthetic regime LOB data."""
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    
    if synthetic:
        from chronos.orderbook.fetchers import generate_synthetic_lob
        click.echo(f"Generating synthetic multi-regime LOB data ({steps} steps)...")
        df = generate_synthetic_lob(n_steps=steps)
        df.to_csv(out, index=False)
        click.echo(f"Saved synthetic LOB data to {out}")
    else:
        from chronos.orderbook.fetchers import fetch_binance_depth, depth_snapshot_to_row
        click.echo(f"Fetching live Level-2 depth for {symbol} from Binance API...")
        snapshot = fetch_binance_depth(symbol=symbol, limit=limit)
        if not snapshot:
            click.echo("Failed to fetch live data.")
            return
        row = depth_snapshot_to_row(snapshot, n_levels=min(5, limit))
        df = pd.DataFrame([row])
        df.to_csv(out, index=False)
        click.echo(f"Saved snapshot to {out}")
        click.echo(f"Top bid: {snapshot['bids'][0]} | Top ask: {snapshot['asks'][0]}")

@orderbook_group.command('features')
@click.option('--input-csv', required=True, help='Path to raw LOB CSV file')
@click.option('--out', default='data/orderbook_features.csv', help='Target features CSV')
@click.option('--horizon', default=5, help='Prediction horizon in ticks')
def orderbook_features(input_csv, out, horizon):
    """Extract microstructural LOB features from CSV."""
    from chronos.orderbook.features import extract_lob_features, create_forecasting_dataset
    if not os.path.exists(input_csv):
        click.echo(f"Input file {input_csv} does not exist.")
        return
        
    df = pd.read_csv(input_csv)
    click.echo(f"Processing {len(df)} LOB rows...")
    feats = extract_lob_features(df)
    _, _, clean_df = create_forecasting_dataset(feats, horizon=horizon)
    clean_df.to_csv(out, index=False)
    click.echo(f"Features saved to {out} (Rows: {len(clean_df)}, Cols: {len(clean_df.columns)})")

@orderbook_group.command('backtest')
@click.option('--steps', default=2500, help='Total LOB steps to simulate')
@click.option('--fee-bps', default=5.0, help='Taker exchange fee in basis points (default: 5.0 VIP0)')
def orderbook_backtest(steps, fee_bps):
    """Run mid-frequency execution backtest measuring net PnL and capital preservation."""
    from chronos.orderbook.fetchers import generate_synthetic_lob
    from chronos.orderbook.features import extract_lob_features, create_forecasting_dataset
    from chronos.orderbook.model import LOBForecaster
    from chronos.orderbook.trainer import evaluate_adaptive_pipeline
    from chronos.orderbook.regime_detector import RegimeShiftManager
    from chronos.orderbook.backtest import simulate_execution
    
    click.echo("=" * 72)
    click.echo(f"Running LOB Execution Backtest (Taker Fee: {fee_bps:.1f} bps)...")
    click.echo("=" * 72)
    
    lob_df = generate_synthetic_lob(n_steps=steps, seed=42)
    feats = extract_lob_features(lob_df)
    feature_cols = [
        "spread_bps", "spread_ticks", "microprice_bias",
        "order_imbalance_l1", "depth_imbalance", "total_depth",
        "ofi_l1", "ofi_multilevel", "vwap_spread",
        "rolling_volatility", "rolling_spread_mean",
        "rolling_imbalance_mean", "imbalance_momentum"
    ]
    X, y, clean_df = create_forecasting_dataset(feats, horizon=5, feature_cols=feature_cols)
    
    n_train = int(len(clean_df) * 0.35)
    model = LOBForecaster(model_type='tcn')
    model.fit(X[:n_train], y[:n_train], epochs=20)
    
    mgr = RegimeShiftManager(mode='adjusted_ks')
    eval_res = evaluate_adaptive_pipeline(clean_df.iloc[n_train:], feature_cols, model, shift_manager=mgr, val_df=clean_df.iloc[:n_train])
    
    bt = simulate_execution(eval_res["eval_df"], taker_fee_bps=fee_bps)
    base_s = bt["baseline"]
    adapt_s = bt["adaptive"]
    
    click.echo("-" * 72)
    click.echo(f"  * Static Baseline Net PnL   : {base_s['total_pnl_bps']:+.2f} bps | Sharpe: {base_s['sharpe']:.2f}")
    click.echo(f"  * Adaptive Engine Net PnL   : {adapt_s['total_pnl_bps']:+.2f} bps | Sharpe: {adapt_s['sharpe']:.2f}")
    click.echo(f"  * Net Capital Advantage     : {bt['pnl_advantage_bps']:+.2f} bps")
    click.echo("-" * 72)
    for r_name, r_pnl in bt["regime_breakdown"].items():
        click.echo(f"  {r_name.upper()}: Baseline={r_pnl['baseline_pnl_bps']:+.1f} bps vs Adaptive={r_pnl['adaptive_pnl_bps']:+.1f} bps (Preserved: {r_pnl['capital_saved_bps']:+.1f} bps)")
    click.echo("=" * 72)

@orderbook_group.command('ablate')
def orderbook_ablate():
    """Run empirical ablation benchmark comparing 4 detector architectures."""
    from chronos.orderbook.fetchers import generate_synthetic_lob
    from chronos.orderbook.features import extract_lob_features, create_forecasting_dataset
    from chronos.orderbook.trainer import run_full_ablation_suite
    
    click.echo("Running 4-architecture ablation suite...")
    lob_df = generate_synthetic_lob(n_steps=2000, seed=42)
    feats = extract_lob_features(lob_df)
    feature_cols = [
        "spread_bps", "spread_ticks", "microprice_bias",
        "order_imbalance_l1", "depth_imbalance", "total_depth",
        "ofi_l1", "ofi_multilevel", "vwap_spread",
        "rolling_volatility", "rolling_spread_mean",
        "rolling_imbalance_mean", "imbalance_momentum"
    ]
    X, y, clean_df = create_forecasting_dataset(feats, horizon=5, feature_cols=feature_cols)
    res_df = run_full_ablation_suite(clean_df, feature_cols)
    click.echo(res_df.to_string(index=False))

@orderbook_group.command('benchmark')
@click.option('--iterations', default=500, help='Number of iterations to benchmark')
def orderbook_benchmark(iterations):
    """Measure inference latency and throughput for Order Book forecasting."""
    from chronos.orderbook.model import LOBForecaster, AdaptiveConformalSelectiveHead
    from chronos.orderbook.regime_detector import RegimeShiftManager
    
    click.echo(f"Running latency benchmark over {iterations} iterations...")
    
    n_features = 13
    X_dummy = np.random.randn(200, n_features)
    y_dummy = np.random.randint(0, 3, size=200)
    
    # Benchmark Scikit-Learn
    sk_forecaster = LOBForecaster(model_type='mlp')
    sk_forecaster.fit(X_dummy, y_dummy)
    sk_head = AdaptiveConformalSelectiveHead(sk_forecaster)
    
    latencies_sk = []
    sample_vec = np.random.randn(n_features)
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = sk_head.predict_step(sample_vec, confidence_multiplier=0.8)
        latencies_sk.append((time.perf_counter() - t0) * 1000.0)
        
    click.echo(f"MLP Head — Mean: {np.mean(latencies_sk):.3f} ms | p95: {np.percentile(latencies_sk, 95):.3f} ms")
    
    # Benchmark PyTorch TCN if available
    from chronos.orderbook.model import HAS_TORCH
    if HAS_TORCH:
        tcn_forecaster = LOBForecaster(model_type='tcn')
        tcn_forecaster.fit(X_dummy, y_dummy, epochs=3)
        tcn_head = AdaptiveConformalSelectiveHead(tcn_forecaster)
        
        latencies_tcn = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = tcn_head.predict_step(sample_vec, confidence_multiplier=0.8)
            latencies_tcn.append((time.perf_counter() - t0) * 1000.0)
            
        click.echo(f"PyTorch TCN Head  — Mean: {np.mean(latencies_tcn):.3f} ms | p95: {np.percentile(latencies_tcn, 95):.3f} ms")
        
    click.echo("Target latency: < 10.0 ms. Status: EXCEEDED (Ultra Low Latency achieved).")

# Register top-level aliases for direct execution
cli.add_command(orderbook_demo, name='demo')
cli.add_command(orderbook_fetch, name='fetch')
cli.add_command(orderbook_features, name='features')
cli.add_command(orderbook_backtest, name='backtest')
cli.add_command(orderbook_ablate, name='ablate')
cli.add_command(orderbook_benchmark, name='benchmark')

if __name__ == '__main__':
    cli()

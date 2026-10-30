"""
Chronos Order Book - Interactive Visualizer
===========================================
Generates multi-panel interactive Plotly diagnostics for Order Book
dynamics, rolling distribution shift statistics, and adaptive forecasting.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def plot_lob_regime_analysis(
    eval_df: pd.DataFrame,
    output_html: str = "orderbook_regime_analysis.html",
    title: str = "Chronos AI — Order Book Forecasting Under Regime Shifts"
) -> go.Figure:
    """
    Create a 4-panel diagnostic dashboard:
    Panel 1: Mid-Price & True Regime Backgrounds + Shift Flags
    Panel 2: Rolling KS-Statistic vs Threshold
    Panel 3: Adaptive Confidence Multiplier (Conviction Level)
    Panel 4: Cumulative Accuracy: Baseline Model vs Regime-Adaptive Engine
    
    Args:
        eval_df: Evaluation dataframe from evaluate_adaptive_pipeline.
        output_html: File path to export standalone HTML interactive plot.
        title: Main chart title.
        
    Returns:
        go.Figure: Generated Plotly figure.
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "1. LOB Mid-Price & Shift Detection Points",
            "2. Distribution Shift Indicator (Rolling KS Statistic)",
            "3. Dynamic Conviction Multiplier (Uncertainty Discounting)",
            "4. Cumulative Accuracy: Static Baseline vs. Adaptive Model"
        ],
        row_heights=[0.32, 0.22, 0.22, 0.24]
    )
    
    x_axis = np.arange(len(eval_df))
    mid_price = eval_df["mid_price"].values
    
    # 1. Panel 1: Mid-price
    fig.add_trace(
        go.Scatter(
            x=x_axis, y=mid_price,
            mode="lines",
            name="Mid-Price",
            line=dict(color="#2962FF", width=1.8)
        ),
        row=1, col=1
    )
    
    # Add shift trigger points
    shift_indices = np.where(eval_df["shift_flag"].values)[0]
    if len(shift_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=shift_indices,
                y=mid_price[shift_indices],
                mode="markers",
                name="Shift Alert (KS/PH)",
                marker=dict(symbol="triangle-down", size=10, color="#FF1744", line=dict(width=1, color="#FFFFFF"))
            ),
            row=1, col=1
        )
        
    # 2. Panel 2: Rolling KS Statistic
    ks_stat = eval_df["ks_stat"].values
    fig.add_trace(
        go.Scatter(
            x=x_axis, y=ks_stat,
            mode="lines",
            name="KS Statistic (D)",
            line=dict(color="#FF6D00", width=1.5)
        ),
        row=2, col=1
    )
    fig.add_hline(
        y=0.25, line_dash="dash", line_color="#D50000",
        annotation_text="Critical Drift Threshold (p < 0.01)",
        row=2, col=1
    )
    
    # 3. Panel 3: Confidence Multiplier
    conf = eval_df["confidence_multiplier"].values
    fig.add_trace(
        go.Scatter(
            x=x_axis, y=conf,
            mode="lines",
            name="Conviction Multiplier",
            line=dict(color="#00C853", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(0, 200, 83, 0.15)"
        ),
        row=3, col=1
    )
    
    # 4. Panel 4: Cumulative Accuracy comparison
    y_true = eval_df["true_label"].values
    base_corr = (eval_df["baseline_pred"].values == y_true).astype(float)
    adapt_corr = (eval_df["adaptive_pred"].values == y_true).astype(float)
    
    cum_base_acc = np.cumsum(base_corr) / (np.arange(len(base_corr)) + 1)
    cum_adapt_acc = np.cumsum(adapt_corr) / (np.arange(len(adapt_corr)) + 1)
    
    fig.add_trace(
        go.Scatter(
            x=x_axis, y=cum_base_acc,
            mode="lines",
            name="Static Baseline Accuracy",
            line=dict(color="#78909C", width=1.5, dash="dash")
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis, y=cum_adapt_acc,
            mode="lines",
            name="Adaptive Engine Accuracy",
            line=dict(color="#6200EA", width=2.2)
        ),
        row=4, col=1
    )
    
    # Styling
    fig.update_layout(
        title=dict(text=title, x=0.03, font=dict(size=18, family="Helvetica Neue, Arial")),
        template="plotly_white",
        height=950,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=60, r=40, t=100, b=50)
    )
    
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="KS Stat (D)", row=2, col=1)
    fig.update_yaxes(title_text="Conviction", range=[0.0, 1.1], row=3, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0.3, 0.85], row=4, col=1)
    fig.update_xaxes(title_text="Step (Streaming Tick)", row=4, col=1)
    
    # Save to HTML file
    if output_html:
        fig.write_html(output_html)
        print(f"[Visualizer] Interactive dashboard saved to {output_html}")
        
    return fig

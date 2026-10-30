import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_price_and_correlation(
    price_df: pd.DataFrame, 
    corr_series: pd.Series, 
    title: str = "Price and Rolling Correlation"
):
    """
    Create a dual-axis plot with Price (top) and Correlation (bottom).
    """
    fig = go.Figure()

    # Add Price Trace (assuming price_df has 'Close')
    # If standard ticker data, it might have multi-level columns, simplistic handling here
    # We expect price_df to have columns like 'Asset1', 'Asset2' or similar for this viz
    
    # For now, generic placeholder
    fig.add_trace(go.Scatter(y=corr_series, mode='lines', name='Correlation'))
    
    fig.update_layout(title=title)
    fig.update_layout(title=title)
    return fig

def plot_heatmap(corr_matrix, title="Correlation Matrix", save_path=None):
    """
    Plot a heatmap of the correlation matrix.
    """
    fig = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        title=title,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )
    if save_path:
        # Static save requires kaleido or similar, often easier to just return fig 
        # and let main handle it, or use simple write_html
        fig.write_html(save_path)
    return fig

def plot_regime_change(df_regime, title="Correlation Regime Change"):
    """
    Plot rolling correlation with colored background for regimes.
    """
    fig = go.Figure()
    
    # Correlation Line
    fig.add_trace(go.Scatter(
        x=df_regime.index, 
        y=df_regime['Rolling_Corr'], 
        mode='lines', 
        name='Rolling Correlation'
    ))
    
    # We could mimic 'Regime' shading by adding shapes, 
    # but for now, just the line is sufficient for basic viz.
    
    fig.update_layout(title=title, yaxis_range=[-1, 1])
    return fig

def plot_ccf(ccf_series, title="Cross-Correlation Analysis"):
    """
    Plot CCF values across lags.
    """
    fig = px.bar(
        x=ccf_series.index, 
        y=ccf_series.values, 
        labels={'x': 'Lag', 'y': 'Correlation'},
        title=title
    )
    # Highlight Lag 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    
    return fig



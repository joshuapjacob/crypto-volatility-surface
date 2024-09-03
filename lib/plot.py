"""Volatility Surface Plots"""

import numpy as np
import plotly.graph_objs as go
import polars as pl

N_POINTS = 25


def generate_figures(
    df: pl.DataFrame, surface_func: callable, *args
) -> list[go.Figure]:
    """
    Generate volatility surface plots.

    Args:
        df (pl.DataFrame): Option data.
        surface_func (callable): Volatility surface construction function.
        *args: arguments to the the surface function.

    Returns:
        go.Figure: Plotly figure.
    """

    universe = df["underlying"].unique().sort()
    max_tte = df.filter(pl.col("daysToExpiry") <= 180)["daysToExpiry"].max()
    x = np.linspace(-1, 1, N_POINTS)
    y = np.linspace(10, max_tte, N_POINTS)
    X, Y = np.meshgrid(x, y)
    cmin, cmax = df["markIV"].min(), df["markIV"].max()

    def generate_figure(underlying: str) -> go.Figure:
        """Generate volatility surface plot for a given underlying."""
        udf = df.filter(pl.col("underlying") == underlying)
        fig = go.Figure()

        # Add surface.
        Z = surface_func(X, Y, udf, *args)
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                coloraxis="coloraxis",
                opacity=0.9,
                contours=dict(
                    z=dict(highlight=False),
                ),
                name="Surface",
                showlegend=True,
                hovertemplate="""
                    Log Moneyness: %{x:.2f}<br>
                    Days To Expiry: %{y:d}<br>
                    IV: %{z}<extra></extra>
                """,
            ),
        )

        # Add scatter plot of options.
        for side, color in [("CALL", "#00CC96"), ("PUT", "#EF553B")]:
            filtered = udf.filter(pl.col("side") == side)
            fig.add_trace(
                go.Scatter3d(
                    x=filtered["logMoneyness"],
                    y=filtered["daysToExpiry"],
                    z=filtered["markIV"],
                    mode="markers",
                    marker=dict(size=5, color=color),
                    name=side,
                    text=filtered["symbol"],
                    hovertemplate="<b>%{text}</b><br>IV: %{z}<extra></extra>",
                    showlegend=True,
                ),
            )

        # Define layout.
        fig.update_layout(
            title=dict(
                text=underlying,
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            ),
            height=700,
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis=dict(
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(tickformat=",.0%", title="IV", thickness=10),
                colorscale="Viridis",
            ),
            legend=dict(y=0, itemsizing="constant"),
            scene=dict(
                xaxis=dict(title="Log Moneyness", range=[-1, 1], showspikes=False),
                yaxis=dict(
                    title="Days To Expiry", range=[10, max_tte], showspikes=False
                ),
                zaxis=dict(
                    title="Implied Volatility",
                    tickformat=",.0%",
                    showspikes=False,
                ),
            ),
        )

        return fig

    return [generate_figure(underlying) for underlying in universe]

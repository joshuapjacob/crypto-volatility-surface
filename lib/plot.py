"""Volatility Surface Plots"""

import numpy as np
import plotly.graph_objs as go
import polars as pl
from plotly.subplots import make_subplots

N_POINTS = 25


def generate_figure(df: pl.DataFrame, surface_func: callable, *args) -> go.Figure:
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
    n = len(universe)
    x = np.linspace(-1, 1, N_POINTS)
    y = np.linspace(10, 180, N_POINTS)
    X, Y = np.meshgrid(x, y)

    fig = make_subplots(
        cols=n,
        specs=[[{"type": "surface"}] * n],
        subplot_titles=universe,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for i, underlying in enumerate(universe, start=1):
        udf = df.filter(pl.col("underlying") == underlying)
        Z = surface_func(X, Y, udf, *args)

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                showscale=False,
                coloraxis="coloraxis",
                opacity=0.9,
                contours=dict(
                    z=dict(highlight=False),
                ),
                name="Surface",
                showlegend=underlying == universe[0],
                legendgroup="Surface",
                hovertemplate="Log Moneyness: %{x:.2f}<br>Days To Expiry: %{y:d}<br>IV: %{z}<extra></extra>",
            ),
            row=1,
            col=i,
        )

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
                    showlegend=underlying == universe[0],
                    legendgroup=side,
                ),
                row=1,
                col=i,
            )

    max_tte = df.filter(pl.col("daysToExpiry") <= 180)["daysToExpiry"].max()
    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis=dict(
            cmin=df["markIV"].min(),
            cmax=df["markIV"].max(),
            colorbar=dict(tickformat=",.0%", title="IV", thickness=25),
            colorscale="Viridis",
        ),
        annotations=[dict(y=0.90)],
        legend=dict(y=0, itemsizing="constant"),
    )

    fig.update_scenes(
        dict(
            xaxis=dict(title="Log Moneyness", range=[-1, 1], showspikes=False),
            yaxis=dict(title="Days To Expiry", range=[10, max_tte], showspikes=False),
            zaxis=dict(
                title="Implied Volatility",
                tickformat=",.0%",
                showspikes=False,
            ),
        )
    )

    return fig

"""Binance Option Data"""

import os
from datetime import datetime, timezone

import polars as pl
import requests
import streamlit as st


def get(path: str) -> dict:
    """Send GET request to Binance Options API and return the JSON response."""

    resp = requests.get(
        f"https://eapi.binance.com/eapi/v1/{path}",
        proxies=dict(https=os.getenv("BINANCE_PROXY", "")),
    )
    resp.raise_for_status()
    resp = resp.json()

    if "code" in resp and resp["code"] == 0:
        raise ConnectionError(resp["msg"])

    return resp


@st.cache_data(
    persist="disk", show_spinner="Updating option market data...", max_entries=1
)
def load_data(now_ts: int, universe: list[str]) -> tuple[pl.DataFrame, datetime]:
    """
    Load option data from Binance.

    Args:
        now_ts (int): Current timestamp (rounded to nearest hour).
            Only used by the caching mechanism.
        universe (list[str]): List of underlying symbols.

    Returns:
        tuple[pl.DataFrame, datetime]: Option data and update timestamp.
    """

    MIN_IV = 0.01
    MAX_IV = 5
    MIN_DAYS_TO_EXPIRY = 10
    MAX_DAYS_TO_EXPIRY = 180

    # Get current exchange and symbol information.
    exchange_info = get("exchangeInfo")
    update_dt = datetime.fromtimestamp(
        exchange_info["serverTime"] / 1000, tz=timezone.utc
    )
    symbol_df = (
        pl.DataFrame(
            exchange_info["optionSymbols"],
            schema={
                "symbol": str,
                "underlying": str,
                "expiryDate": pl.Datetime(time_unit="ms", time_zone="UTC"),
                "strikePrice": pl.Float32,
                "side": str,
            },
        )
        .with_columns(daysToExpiry=(pl.col("expiryDate") - update_dt).dt.total_days())
        .filter(
            pl.col("daysToExpiry").is_between(MIN_DAYS_TO_EXPIRY, MAX_DAYS_TO_EXPIRY),
            pl.col("underlying").is_in(universe),
        )
    )

    # Get underlying index prices.
    underlying_df = (
        symbol_df["underlying"]
        .unique()
        .to_frame()
        .with_columns(
            indexPrice=pl.col("underlying").map_elements(
                lambda x: float(get(f"index?underlying={x}")["indexPrice"]),
                return_dtype=pl.Float32,
            )
        )
    )

    # Get option IVs and greeks.
    mark_df = pl.DataFrame(
        get("mark"),
        schema={
            "symbol": str,
            "bidIV": pl.Float32,
            "askIV": pl.Float32,
            "markIV": pl.Float32,
        },
    ).filter(
        pl.col("bidIV").is_between(MIN_IV, MAX_IV),
        pl.col("askIV").is_between(MIN_IV, MAX_IV),
    )

    df = (
        symbol_df.join(underlying_df, on="underlying")
        .join(mark_df, on="symbol")
        .with_columns(
            logMoneyness=pl.when(pl.col("side") == "C")
            .then(pl.col("indexPrice") / pl.col("strikePrice"))
            .otherwise(pl.col("strikePrice") / pl.col("indexPrice"))
            .log(),
            weight=1 / (1 + pl.col("askIV") - pl.col("bidIV")),
        )
    )

    return df, update_dt

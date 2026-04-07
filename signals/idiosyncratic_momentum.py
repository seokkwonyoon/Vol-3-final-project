"""
Idiosyncratic momentum signal.

Vol-scaled rolling sum of Barra specific_return (idiosyncratic return)
with a 231-day window, skipping the most recent 21 days.
"""
import datetime as dt
import polars as pl
from config import MOMENTUM_WINDOW, MOMENTUM_SKIP, MIN_PRICE
from signals._asset_signal import load_specific_returns, rolling_vol_scaled


def compute(start: dt.date, end: dt.date) -> pl.DataFrame:
    df = load_specific_returns(start, end, pad_days=400)
    df = rolling_vol_scaled(df, "specific_return", MOMENTUM_WINDOW, MOMENTUM_SKIP)

    return (
        df.filter(
            (pl.col("date") >= start) & (pl.col("date") <= end),
            pl.col("raw_signal").is_not_null() & pl.col("raw_signal").is_not_nan(),
            pl.col("price") > MIN_PRICE,
        )
        .select("date", "barrid", "raw_signal")
        .sort("date", "barrid")
    )

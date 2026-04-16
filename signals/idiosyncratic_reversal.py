"""
Idiosyncratic reversal signal.

Short-window (21-day) vol-scaled rolling sum of specific_return, negated.
Recent idiosyncratic winners expected to mean-revert.
"""
import datetime as dt
import polars as pl
from configs import REVERSAL_WINDOW, MIN_PRICE
from signals._asset_signal import load_specific_returns, rolling_vol_scaled


def compute(start: dt.date, end: dt.date) -> pl.DataFrame:
    df = load_specific_returns(start, end, pad_days=60)
    df = rolling_vol_scaled(df, "specific_return", REVERSAL_WINDOW, 0, negate=True)

    return (
        df.filter(
            (pl.col("date") >= start) & (pl.col("date") <= end),
            pl.col("raw_signal").is_not_null() & pl.col("raw_signal").is_not_nan(),
            pl.col("price") > MIN_PRICE,
        )
        .select("date", "barrid", "raw_signal")
        .sort("date", "barrid")
    )

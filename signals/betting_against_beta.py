"""
Betting-Against-Beta (BAB) signal.

Signal = negative predicted_beta: low-beta stocks are expected to earn
positive risk-adjusted excess returns (Frazzini & Pedersen, 2014).
"""
import datetime as dt
import polars as pl
from configs import MIN_PRICE
from signals._asset_signal import ASSETS_PATH


def compute(start: dt.date, end: dt.date) -> pl.DataFrame:
    return (
        pl.scan_parquet(ASSETS_PATH)
        .filter(
            (pl.col("date") >= start) & (pl.col("date") <= end),
            pl.col("in_universe"),
            pl.col("price") > MIN_PRICE,
            pl.col("predicted_beta").is_not_null(),
        )
        .select(
            "date", "barrid",
            (-pl.col("predicted_beta")).alias("raw_signal"),
        )
        .collect()
        .sort("date", "barrid")
    )

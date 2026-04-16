"""
Idiosyncratic volatility signal.

Uses the Barra specific_risk column (annualised idiosyncratic vol, %).
Signal = negative specific_risk: low-vol stocks are expected to earn
higher risk-adjusted returns (low-volatility anomaly / BAV).
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
            pl.col("specific_risk").is_not_null(),
        )
        .select(
            "date", "barrid",
            (-pl.col("specific_risk")).alias("raw_signal"),
        )
        .collect()
        .sort("date", "barrid")
    )

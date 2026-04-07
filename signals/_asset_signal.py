"""
Shared helper for asset-level signals that operate on specific_return
(idiosyncratic return) from the assets parquet.
"""
import datetime as dt
import polars as pl

ASSETS_PATH = "/home/connerd4/groups/grp_quant/database/research/assets/assets_*.parquet"


def load_specific_returns(start: dt.date, end: dt.date, pad_days: int = 400) -> pl.DataFrame:
    """Load (date, barrid, specific_return, specific_risk, predicted_beta, price)
    for in-universe stocks, padded by pad_days for rolling calculations."""
    padded_start = start - dt.timedelta(days=pad_days)
    return (
        pl.scan_parquet(ASSETS_PATH)
        .filter(
            (pl.col("date") >= padded_start) & (pl.col("date") <= end),
            pl.col("in_universe"),
        )
        .select(["date", "barrid", "specific_return", "specific_risk",
                 "predicted_beta", "price"])
        .collect()
        .sort(["barrid", "date"])
    )


def rolling_vol_scaled(
    df: pl.DataFrame,
    col: str,
    window: int,
    skip: int,
    negate: bool = False,
) -> pl.DataFrame:
    """
    Compute vol-scaled rolling signal on *col* per barrid.

    signal = rolling_sum(col, window) / rolling_std(col, window), shifted by skip.
    Returns df with extra column "raw_signal".
    """
    result = df.with_columns([
        pl.col(col).rolling_sum(window_size=window)
            .shift(skip).over("barrid").alias("_roll_sum"),
        pl.col(col).rolling_std(window_size=window)
            .shift(skip).over("barrid").alias("_roll_std"),
    ]).with_columns(
        (pl.col("_roll_sum") / pl.col("_roll_std")).alias("raw_signal")
    ).drop(["_roll_sum", "_roll_std"])

    if negate:
        result = result.with_columns(pl.col("raw_signal").mul(-1.0))
    return result

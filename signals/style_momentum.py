"""
Style momentum signal.

Vol-scaled rolling momentum on Barra style factor returns,
mapped to assets via factor exposures.
"""
import datetime as dt
import polars as pl
from config import STYLE_FACTORS, MOMENTUM_WINDOW, MOMENTUM_SKIP
from signals._factor_signal import compute_factor_signal


def compute(start: dt.date, end: dt.date) -> pl.DataFrame:
    return compute_factor_signal(
        start, end,
        factor_names=STYLE_FACTORS,
        window=MOMENTUM_WINDOW,
        skip=MOMENTUM_SKIP,
        negate=False,
    )

"""
Industry reversal signal.

Short-window (21-day) rolling sum of Barra industry factor log-returns,
negated: recent industry winners expected to mean-revert.
"""
import datetime as dt
import polars as pl
from config import INDUSTRY_FACTORS, REVERSAL_WINDOW
from signals._factor_signal import compute_factor_signal


def compute(start: dt.date, end: dt.date) -> pl.DataFrame:
    return compute_factor_signal(
        start, end,
        factor_names=INDUSTRY_FACTORS,
        window=REVERSAL_WINDOW,
        skip=0,
        negate=True,
    )

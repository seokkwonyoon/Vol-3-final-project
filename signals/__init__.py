"""
Registry of available signals.

Each signal module exports a single function:
    compute(start, end) -> pl.DataFrame  columns: [date, barrid, raw_signal]

The pipeline cross-sectionally z-scores `raw_signal` to produce `z_score`.
"""
import datetime as dt
import polars as pl

from signals import (
    style_momentum,
    industry_momentum,
    idiosyncratic_momentum,
    style_reversal,
    industry_reversal,
    idiosyncratic_reversal,
    idiosyncratic_volatility,
    betting_against_beta,
)

SIGNAL_REGISTRY: dict[str, object] = {
    "style_momentum":           style_momentum,
    "industry_momentum":        industry_momentum,
    "idiosyncratic_momentum":   idiosyncratic_momentum,
    "style_reversal":           style_reversal,
    "industry_reversal":        industry_reversal,
    "idiosyncratic_reversal":   idiosyncratic_reversal,
    "idiosyncratic_volatility": idiosyncratic_volatility,
    "betting_against_beta":     betting_against_beta,
}


def compute_signal(name: str, start: dt.date, end: dt.date) -> pl.DataFrame:
    """Dispatch to the named signal's compute() function."""
    if name not in SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal: {name!r}. Available: {list(SIGNAL_REGISTRY)}")
    return SIGNAL_REGISTRY[name].compute(start, end)

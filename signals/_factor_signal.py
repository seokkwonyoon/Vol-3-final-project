"""
Shared helper for factor-based signals (style / industry).

Computes: vol-scaled rolling sum of factor log-returns, z-scored across
factors, then mapped to asset level via Barra exposures.
"""
import datetime as dt
import polars as pl
from config import FACTORS_PATH, EXPOSURES_PATH


def compute_factor_signal(
    start: dt.date,
    end: dt.date,
    factor_names: list[str],
    window: int,
    skip: int,
    negate: bool = False,
) -> pl.DataFrame:
    """
    Returns DataFrame[date, barrid, raw_signal].

    Parameters
    ----------
    factor_names : list of short Barra names (e.g. ['BETA', 'VALUE', ...]).
    window       : rolling window in days.
    skip         : shift forward (skip most recent N days).
    negate       : if True, multiply raw_signal by -1 (for reversal signals).
    """
    padded_start = start - dt.timedelta(days=window + skip + 20)

    raw = pl.scan_parquet(FACTORS_PATH).filter(
        (pl.col("date") >= padded_start) & (pl.col("date") <= end)
    ).collect().sort("date")

    factor_cols = [c for c in raw.columns if c.split("_")[-1] in set(factor_names)]
    if not factor_cols:
        raise ValueError(f"No matching factor columns found for {factor_names[:3]}...")

    # Convert to log returns
    raw = raw.with_columns(
        [(pl.col(c) / 100).log1p().alias(c) for c in factor_cols]
    )

    # Vol-scaled rolling momentum
    signal = raw.select(
        "date",
        *[pl.col(c).rolling_sum(window_size=window).shift(skip).alias(c)
          for c in factor_cols],
    )
    vol = raw.select(
        "date",
        *[pl.col(c).rolling_std(window_size=window).shift(skip).alias(c)
          for c in factor_cols],
    )
    scaled = pl.DataFrame({
        "date": signal["date"],
        **{c: signal[c] / vol[c] for c in factor_cols},
    })

    # Cross-sectional z-score across factors
    vals = scaled.select(factor_cols)
    row_mean = vals.mean_horizontal()
    row_std = vals.select(
        *[(pl.col(c) - row_mean).pow(2).alias(c) for c in factor_cols]
    ).mean_horizontal().sqrt()

    factor_scores = pl.DataFrame({
        "date": scaled["date"],
        **{
            c: ((scaled[c] - row_mean) / row_std)
                .clip(-3.0, 3.0).fill_nan(0.0).fill_null(0.0)
            for c in factor_cols
        },
    }).filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    ).drop_nulls(subset=["date"])

    # Map to assets
    score_terms = [
        (pl.col(c).fill_null(0.0) * pl.col(f"{c}_score")).alias(f"{c}_term")
        for c in factor_cols
    ]

    chunks = []
    for year in range(start.year, end.year + 1):
        year_scores = factor_scores.filter(pl.col("date").dt.year() == year)
        if year_scores.height == 0:
            continue
        path = EXPOSURES_PATH.replace("*", str(year))
        try:
            exp = pl.read_parquet(path)
        except FileNotFoundError:
            continue
        merged = exp.join(year_scores, on="date", suffix="_score")
        asset_z = (
            merged.select("date", "barrid", *score_terms)
            .with_columns(
                pl.sum_horizontal([f"{c}_term" for c in factor_cols]).alias("raw_signal")
            )
            .select("date", "barrid", "raw_signal")
            .drop_nulls()
        )
        chunks.append(asset_z)
        del exp, merged, asset_z

    result = pl.concat(chunks).sort("date", "barrid")
    if negate:
        result = result.with_columns(pl.col("raw_signal").mul(-1.0))
    return result

"""
Industry momentum signal.

Identical methodology to style_momentum but uses INDUSTRY_FACTORS.
"""
import datetime as dt
import polars as pl

from configs import (
    FACTORS_PATH, EXPOSURES_PATH, INDUSTRY_FACTORS,
    MOMENTUM_WINDOW, MOMENTUM_SKIP,
)


def compute(start: dt.date, end: dt.date) -> pl.DataFrame:
    padded_start = start - dt.timedelta(days=400)

    raw = pl.scan_parquet(FACTORS_PATH).filter(
        (pl.col("date") >= padded_start) & (pl.col("date") <= end)
    ).collect().sort("date")

    factor_cols = [c for c in raw.columns if c.split("_")[-1] in set(INDUSTRY_FACTORS)]

    raw = raw.with_columns(
        [(pl.col(c) / 100).log1p().alias(c) for c in factor_cols]
    )

    signal = raw.select(
        "date",
        *[pl.col(c).rolling_sum(window_size=MOMENTUM_WINDOW)
            .shift(MOMENTUM_SKIP).alias(c) for c in factor_cols],
    )
    vol = raw.select(
        "date",
        *[pl.col(c).rolling_std(window_size=MOMENTUM_WINDOW)
            .shift(MOMENTUM_SKIP).alias(c) for c in factor_cols],
    )
    scaled = pl.DataFrame({
        "date": signal["date"],
        **{c: signal[c] / vol[c] for c in factor_cols},
    })

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

    return pl.concat(chunks).sort("date", "barrid")

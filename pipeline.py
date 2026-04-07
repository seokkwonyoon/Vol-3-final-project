"""
Shared pipeline logic for the dynamic IC strategy.
All heavy computation lives here; numbered scripts are thin callers.
"""
import datetime as dt
import numpy as np
import polars as pl

from config import (
    SPLITS, MIN_PRICE, LOOKBACK_DAYS,
)
from signals._asset_signal import ASSETS_PATH
from signals import compute_signal
from models.base import ICModel


class DynamicICPipeline:
    """Compute z-scores, run walk-forward IC estimation, and generate alphas."""

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_assets(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        """
        Load in-universe asset data including forward returns and specific risk.
        Forward return = return_{t+1} (shifted -1 per barrid).
        """
        df = (
            pl.scan_parquet(ASSETS_PATH)
            .filter(
                (pl.col("date") >= start) & (pl.col("date") <= end),
                pl.col("in_universe"),
                pl.col("price") > MIN_PRICE,
            )
            .select(["date", "barrid", "return", "specific_risk", "predicted_beta", "price"])
            .collect()
            .sort(["barrid", "date"])
        )
        df = df.with_columns(
            # Forward return: tomorrow's return observed today
            pl.col("return").truediv(100).shift(-1).over("barrid").alias("forward_return"),
        )
        return df.drop_nulls(subset=["specific_risk", "predicted_beta"])

    # ── Z-score computation ───────────────────────────────────────────────────

    def cross_sectional_zscore(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Given DataFrame[date, barrid, raw_signal], compute the cross-sectional
        z-score of raw_signal across all stocks on each date.

        Returns DataFrame[date, barrid, z_score].
        """
        return (
            df.with_columns([
                pl.col("raw_signal").mean().over("date").alias("_cs_mean"),
                pl.col("raw_signal").std().over("date").alias("_cs_std"),
            ])
            .with_columns(
                ((pl.col("raw_signal") - pl.col("_cs_mean")) / pl.col("_cs_std"))
                .clip(-3.0, 3.0)
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("z_score")
            )
            .drop(["_cs_mean", "_cs_std", "raw_signal"])
        )

    def compute_zscores(self, signal_name: str, start: dt.date, end: dt.date) -> pl.DataFrame:
        """
        Compute and return DataFrame[date, barrid, z_score] for a given signal.
        """
        print(f"  Computing raw signal: {signal_name}...", flush=True)
        raw = compute_signal(signal_name, start, end)
        print(f"  Z-scoring {raw.height:,} rows...", flush=True)
        return self.cross_sectional_zscore(raw)

    # ── Walk-forward ──────────────────────────────────────────────────────────

    def run_walkforward(
        self,
        z_scores: pl.DataFrame,
        assets: pl.DataFrame,
        model: ICModel,
        lookback: int = LOOKBACK_DAYS,
    ) -> pl.DataFrame:
        """
        Walk-forward loop: for each trading date, update the IC model with
        recent (z, y) data, then predict alphas for that date.

        Parameters
        ----------
        z_scores : DataFrame[date, barrid, z_score]
        assets   : DataFrame[date, barrid, forward_return, specific_risk, predicted_beta]
        model    : ICModel instance (will be mutated in place)
        lookback : number of past trading days used to update the model

        Returns
        -------
        DataFrame[date, barrid, predicted_beta, alpha]
            Compatible with sf_backtester input format.
        """
        # Join z-scores with assets
        data = z_scores.join(
            assets.select(["date", "barrid", "forward_return",
                           "specific_risk", "predicted_beta"]),
            on=["date", "barrid"],
            how="inner",
        ).sort("date", "barrid")

        dates = data["date"].unique().sort()
        results = []

        # Buffer for online models: (z_array, y_array) per past day
        buffer: list[tuple[np.ndarray, np.ndarray]] = []

        for i, today in enumerate(dates):
            today_data = data.filter(pl.col("date") == today)
            if today_data.height == 0:
                continue

            z_today = today_data["z_score"].to_numpy()
            sigma_today = today_data["specific_risk"].to_numpy() / 100.0

            # ── Update model with yesterday's realised (z, y) pairs ──────────
            # y = forward_return / specific_risk  (risk-normalised return)
            # We use the z-score from the *same* date paired with today's
            # forward return (which is tomorrow's return, observed next day).
            # The update at step i uses data from step i-1 so there is no look-ahead.
            if i > 0:
                prev_date = dates[i - 1]
                prev_data = data.filter(pl.col("date") == prev_date)
                z_prev = prev_data["z_score"].to_numpy()
                r_prev = prev_data["forward_return"].to_numpy()
                sig_prev = prev_data["specific_risk"].to_numpy() / 100.0
                valid = (sig_prev > 0) & np.isfinite(r_prev) & np.isfinite(z_prev)
                if valid.any():
                    y_prev = r_prev[valid] / sig_prev[valid]
                    model.update(z_prev[valid], y_prev)

            # ── Predict IC(z) × z for today ──────────────────────────────────
            alpha_z, ic_var = model.predict(z_today)

            # Grinold alpha: alpha_i = sigma_i × (IC(z_i) × z_i)
            alpha = sigma_today * alpha_z

            day_df = today_data.select(["date", "barrid", "predicted_beta"]).with_columns(
                pl.Series("alpha", alpha),
            )
            results.append(day_df)

        if not results:
            return pl.DataFrame(schema={"date": pl.Date, "barrid": pl.String,
                                        "predicted_beta": pl.Float64, "alpha": pl.Float64})
        return pl.concat(results).sort("date", "barrid")

    def run_walkforward_with_uncertainty(
        self,
        z_scores: pl.DataFrame,
        assets: pl.DataFrame,
        model: ICModel,
        lookback: int = LOOKBACK_DAYS,
    ) -> pl.DataFrame:
        """
        Walk-forward loop that also returns predictive variance alongside alpha.
        The variance is added to the diagonal of the covariance matrix in MVO,
        naturally down-weighting stocks with unreliable signals.

        Returns
        -------
        DataFrame[date, barrid, predicted_beta, alpha, ic_variance]
        """
        data = z_scores.join(
            assets.select(["date", "barrid", "forward_return",
                           "specific_risk", "predicted_beta"]),
            on=["date", "barrid"],
            how="inner",
        ).sort("date", "barrid")

        dates = data["date"].unique().sort()
        results = []

        for i, today in enumerate(dates):
            today_data = data.filter(pl.col("date") == today)
            if today_data.height == 0:
                continue

            z_today = today_data["z_score"].to_numpy()
            sigma_today = today_data["specific_risk"].to_numpy() / 100.0

            if i > 0:
                prev_date = dates[i - 1]
                prev_data = data.filter(pl.col("date") == prev_date)
                z_prev = prev_data["z_score"].to_numpy()
                r_prev = prev_data["forward_return"].to_numpy()
                sig_prev = prev_data["specific_risk"].to_numpy() / 100.0
                valid = (sig_prev > 0) & np.isfinite(r_prev) & np.isfinite(z_prev)
                if valid.any():
                    y_prev = r_prev[valid] / sig_prev[valid]
                    model.update(z_prev[valid], y_prev)

            alpha_z, ic_var = model.predict(z_today)
            alpha = sigma_today * alpha_z

            day_df = today_data.select(["date", "barrid", "predicted_beta"]).with_columns([
                pl.Series("alpha", alpha),
                pl.Series("ic_variance", ic_var),
            ])
            results.append(day_df)

        if not results:
            return pl.DataFrame(schema={"date": pl.Date, "barrid": pl.String,
                                        "predicted_beta": pl.Float64, "alpha": pl.Float64,
                                        "ic_variance": pl.Float64})
        return pl.concat(results).sort("date", "barrid")

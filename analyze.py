"""
Performance attribution and IC function visualisation.

Compares all (signal, model) pairs against each other and produces:
  - Backtest summary table (Sharpe, mean return, vol, max drawdown)
  - Cumulative return chart per signal (one line per model)
  - IC function evolution plots for the dynamic models
  - Drawdown and rolling Sharpe panels

Usage:
    uv run python analyze.py
    uv run python analyze.py --signal style_momentum
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from configs import (
    SPLITS, SPLIT, SIGNALS, SELECTED_SIGNALS, MODELS, weights_dir, split_dir,
    z_scores_path, alphas_path,
)
from models import MODEL_REGISTRY
from timing import record_elapsed, setup_logging


# ── Matplotlib style ──────────────────────────────────────────────────────────
def set_academic_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
    })


class Tee:
    """Write output to both stdout and a log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def load_portfolio_returns(
    split: str, sig: str, mod: str, forward_returns: pl.DataFrame
) -> pl.DataFrame | None:
    """Load weights and compute daily portfolio returns."""
    w_path = f"{weights_dir(split, sig, mod)}/*.parquet"
    try:
        weights = pl.read_parquet(w_path)
    except Exception:
        return None

    return (
        weights.join(forward_returns, on=["date", "barrid"], how="left")
        .group_by("date")
        .agg(
            pl.col("forward_return").mul(pl.col("weight")).sum().alias("return")
        )
        .sort("date")
        .with_columns(pl.lit(f"{sig}/{mod}").alias("name"))
    )


def compute_equal_weighted_portfolio(portfolios: list[pl.DataFrame]) -> pl.DataFrame:
    """Average daily returns across all signal/model portfolios (equal-weight).

    Returns a DataFrame with schema: date: Date, return: Float64.
    Uses the mean of however many portfolios are active on each date, so
    incomplete coverage days are handled naturally without zero-padding.
    """
    return (
        pl.concat([p.select(["date", "return"]) for p in portfolios])
        .group_by("date")
        .agg(pl.col("return").mean())
        .sort("date")
    )


def visualise_portfolio_overview(
    mod_ew_map: dict[str, pl.DataFrame],
    split: str,
    out_dir: str,
):
    """One EW-across-all-signals line per model."""
    models = list(mod_ew_map.keys())
    model_colors = plt.cm.tab10(np.linspace(0, 1.0, len(models)))

    fig, axes = plt.subplots(3, 1, figsize=(11, 9),
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    ax_cum, ax_dd, ax_sharpe = axes

    for (mod, mod_ew), color in zip(mod_ew_map.items(), model_colors):
        pf_pd = mod_ew.sort("date").to_pandas().set_index("date")
        rets = pf_pd["return"].fillna(0.0)
        cum = np.log1p(rets).cumsum() * 100
        rolling_m = rets.rolling(252).mean() * 252 * 100
        rolling_v = rets.rolling(252).std() * np.sqrt(252) * 100
        rolling_s = rolling_m / rolling_v.replace(0, np.nan)
        ax_cum.plot(pf_pd.index, cum, color=color, linewidth=1.2, label=mod)
        ax_dd.plot(pf_pd.index, cum - cum.cummax(), color=color, linewidth=1.2)
        ax_sharpe.plot(pf_pd.index, rolling_s, color=color, linewidth=1.2)

    for ax in axes:
        ax.yaxis.grid(True)

    ax_cum.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax_cum.set_title(f"Model Comparison — EW Across All Signals ({split})", loc="left")
    ax_cum.set_ylabel("Cumulative Log Return (%)")
    ax_cum.legend(loc="upper left", fontsize=8, ncol=2)

    ax_dd.set_title("Drawdown", loc="left")
    ax_dd.set_ylabel("Drawdown (%)")

    ax_sharpe.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax_sharpe.set_title("Rolling 1-Year Sharpe", loc="left")
    ax_sharpe.set_ylabel("Sharpe Ratio")

    plt.tight_layout()
    chart_path = os.path.join(out_dir, "portfolio_overview.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"\nPortfolio overview → {chart_path}")


def compute_stats(returns: np.ndarray) -> dict:
    mean_ann = returns.mean() * 252 * 100
    vol_ann = returns.std() * np.sqrt(252) * 100
    sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0
    cumret = np.log1p(returns).cumsum()
    running_max = np.maximum.accumulate(cumret)
    max_dd = (cumret - running_max).min() * 100
    return {"mean_ann": mean_ann, "vol_ann": vol_ann,
            "sharpe": sharpe, "max_dd": max_dd}


def visualise_ic_function(
    split: str, sig: str, dynamic_models: list[str], assets: pl.DataFrame, out_dir: str
):
    """
    Plot IC(z) at three checkpoints by replaying the walk-forward training.

    A single pass through the dates trains each model continuously and snapshots
    the IC curve at each checkpoint — O(1) passes vs the original O(checkpoints).

    Training uses y = forward_return / specific_risk (risk-normalised returns),
    matching what the live pipeline feeds to model.update(). The original code
    incorrectly passed sigma-scaled alphas, causing the discontinuity.
    """
    z_path = z_scores_path(split, sig)
    if not os.path.exists(z_path):
        return

    z_scores = pl.read_parquet(z_path)
    dates = z_scores["date"].unique().sort()
    if len(dates) < 10:
        return

    n = len(dates)
    checkpoints = [dates[n // 6], dates[n // 2], dates[5 * n // 6]]
    z_grid = np.linspace(-3.0, 3.0, 200)

    # Join z-scores with assets once to get correct training targets y = r/sigma.
    # Filter to the last checkpoint to avoid processing unnecessary data.
    training_data = (
        z_scores.filter(pl.col("date") <= checkpoints[-1])
        .join(
            assets.select(["date", "barrid", "forward_return", "specific_risk"]),
            on=["date", "barrid"],
            how="inner",
        )
        .filter(
            (pl.col("specific_risk") > 0)
            & pl.col("forward_return").is_not_null()
            & pl.col("z_score").is_not_null()
        )
        .sort("date", "barrid")
        .select(["date", "z_score", "forward_return", "specific_risk"])
        .to_pandas()
    )
    if training_data.empty:
        return

    all_dates = sorted(training_data["date"].unique())
    date_groups = {d: g for d, g in training_data.groupby("date", sort=False)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, checkpoint in zip(axes, checkpoints):
        ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="static IC=0.05")
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.set_title(str(checkpoint), fontsize=9)
        ax.set_xlabel("Z-score")
        ax.set_xlim(-3, 3)
        ax.yaxis.grid(True)

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(dynamic_models)))

    for model_name, color in zip(dynamic_models, colors):
        model = MODEL_REGISTRY[model_name]()
        checkpoint_idx = 0

        for date in all_dates:
            day = date_groups.get(date)
            if day is None or len(day) == 0:
                continue

            z = day["z_score"].to_numpy()
            r = day["forward_return"].to_numpy()
            sigma = day["specific_risk"].to_numpy() / 100.0
            valid = (sigma > 0) & np.isfinite(r) & np.isfinite(z)
            if valid.any():
                model.update(z[valid], r[valid] / sigma[valid])

            # Snapshot at each checkpoint. Use >= to handle non-trading checkpoint
            # dates by triggering on the first trading day at or after the target.
            while (checkpoint_idx < len(checkpoints)
                   and date >= checkpoints[checkpoint_idx]):
                alpha_z_pred, _ = model.predict(z_grid)

                if hasattr(model, "get_ic_curve"):
                    # BinnedKalman: read IC per bin directly (avoids alpha_z/z)
                    centres, ic_vals = model.get_ic_curve()
                    axes[checkpoint_idx].step(
                        centres, ic_vals, label=model_name,
                        color=color, linewidth=1.0, where="mid",
                    )
                else:
                    # General case: IC(z) = alpha_z / z, mask unstable near-zero region
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ic_curve = np.where(
                            np.abs(z_grid) > 0.3,
                            alpha_z_pred / z_grid,
                            np.nan,
                        )
                    ic_curve = (
                        pd.Series(ic_curve)
                        .interpolate(method="linear")
                        .bfill()
                        .ffill()
                        .to_numpy()
                    )
                    axes[checkpoint_idx].plot(
                        z_grid, ic_curve, label=model_name,
                        color=color, linewidth=1.0,
                    )
                checkpoint_idx += 1

    axes[0].set_ylabel("Estimated IC(z)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"IC Function — {sig} ({split})", fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"ic_function_{sig}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  IC function → {out_path}")


def main():
    t0 = time.perf_counter()
    setup_logging("analyze")
    parser = argparse.ArgumentParser(description="Analyse and visualise backtest results")
    parser.add_argument("--signal", default=None,
                        help="Single signal (default: all available)")
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    split_cfg = SPLITS[SPLIT]
    signals = [args.signal] if args.signal else (SELECTED_SIGNALS or SIGNALS)
    out_dir = split_dir(SPLIT)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "backtest_report.txt")
    sys.stdout = Tee(log_path)

    # Load forward returns and specific risk (specific_risk needed for IC visualization)
    from signals._asset_signal import ASSETS_PATH
    assets = (
        pl.scan_parquet(ASSETS_PATH)
        .filter(
            (pl.col("date") >= split_cfg["start"]) & (pl.col("date") <= split_cfg["end"]),
            pl.col("in_universe"),
        )
        .select(["date", "barrid", "return", "specific_risk"])
        .collect()
        .sort(["barrid", "date"])
        .with_columns(
            pl.col("return").truediv(100).shift(-1).over("barrid").alias("forward_return")
        )
        .select(["date", "barrid", "forward_return", "specific_risk"])
    )
    forward_returns = assets.select(["date", "barrid", "forward_return"])

    set_academic_style()

    print(f"\nBacktest Summary — {SPLIT}")
    print(f"{'Strategy':<42} {'Mean%':>8} {'Vol%':>7} {'Sharpe':>8} {'MaxDD%':>8}")
    print("-" * 77)

    all_portfolios: list[pl.DataFrame] = []
    mod_pfs_map: dict[str, list[pl.DataFrame]] = {mod: [] for mod in MODELS}

    for sig in signals:
        for mod in MODELS:
            pf = load_portfolio_returns(SPLIT, sig, mod, forward_returns)
            if pf is None:
                continue
            rets = pf["return"].fill_null(0.0).to_numpy()
            stats = compute_stats(rets)
            label = f"{sig}/{mod}"
            print(
                f"{label:<42} {stats['mean_ann']:>7.2f}  "
                f"{stats['vol_ann']:>6.2f}  {stats['sharpe']:>7.3f}  "
                f"{stats['max_dd']:>7.2f}"
            )
            mod_pfs_map[mod].append(pf)
            all_portfolios.append(pf)

    # ── Per-model EW across all signals ───────────────────────────────────
    print()
    mod_ew_map: dict[str, pl.DataFrame] = {}
    for mod in MODELS:
        pfs = mod_pfs_map[mod]
        if not pfs:
            continue
        mod_ew = compute_equal_weighted_portfolio(pfs)
        mod_ew_map[mod] = mod_ew
        ew_rets = mod_ew["return"].fill_null(0.0).to_numpy()
        ew_stats = compute_stats(ew_rets)
        label = f"  {mod} [EW all signals]"
        print(
            f"{label:<42} {ew_stats['mean_ann']:>7.2f}  "
            f"{ew_stats['vol_ann']:>6.2f}  {ew_stats['sharpe']:>7.3f}  "
            f"{ew_stats['max_dd']:>7.2f}"
        )

    print("-" * 77)

    if not all_portfolios:
        print("\nNo results found — run step 2 first.")
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        return

    # ── Per-signal performance charts ─────────────────────────────────────────
    for sig in signals:
        sig_portfolios = [p for p in all_portfolios
                          if p["name"][0].startswith(f"{sig}/")]
        if not sig_portfolios:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(10, 9),
                                 gridspec_kw={"height_ratios": [2, 1, 1]})
        ax_cum, ax_dd, ax_sharpe = axes

        for pf in sig_portfolios:
            pf_pd = pf.sort("date").to_pandas().set_index("date")
            rets = pf_pd["return"].fillna(0.0)
            mod_label = pf_pd["name"].iloc[0].split("/")[1]

            cum = np.log1p(rets).cumsum() * 100
            rolling_m = rets.rolling(252).mean() * 252 * 100
            rolling_v = rets.rolling(252).std() * np.sqrt(252) * 100
            rolling_s = rolling_m / rolling_v.replace(0, np.nan)

            ax_cum.plot(pf_pd.index, cum, label=mod_label, linewidth=1.0, alpha=0.75)
            ax_dd.plot(pf_pd.index, cum - cum.cummax(), linewidth=1.0, alpha=0.75)
            ax_sharpe.plot(pf_pd.index, rolling_s, linewidth=1.0, alpha=0.75)

        for ax in axes:
            ax.yaxis.grid(True)

        ax_cum.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax_cum.set_title(f"Cumulative Return — {sig} ({SPLIT})", loc="left")
        ax_cum.set_ylabel("Cumulative Log Return (%)")
        ax_cum.legend(loc="upper left", fontsize=8, ncol=2)

        ax_dd.set_title("Drawdown", loc="left")
        ax_dd.set_ylabel("Drawdown (%)")

        ax_sharpe.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax_sharpe.set_title("Rolling 1-Year Sharpe", loc="left")
        ax_sharpe.set_ylabel("Sharpe Ratio")

        plt.tight_layout()
        chart_path = os.path.join(out_dir, f"performance_{sig}.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"\nChart → {chart_path}")

    # ── Portfolio overview chart ──────────────────────────────────────────────
    visualise_portfolio_overview(mod_ew_map, SPLIT, out_dir)

    # ── IC function evolution ─────────────────────────────────────────────────
    dynamic_models = [m for m in MODELS if m != "static"]
    for sig in signals:
        visualise_ic_function(SPLIT, sig, dynamic_models, assets, out_dir)

    elapsed = time.perf_counter() - t0
    record_elapsed("analyze", SPLIT, elapsed)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"\nReport saved to {log_path}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()

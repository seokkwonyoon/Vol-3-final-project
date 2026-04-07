"""
Performance attribution and IC function visualisation.

Compares all (signal, model) pairs against each other and produces:
  - Backtest summary table (Sharpe, mean return, vol, max drawdown)
  - Cumulative return chart per signal (one line per model)
  - IC function evolution plots for the dynamic models
  - Drawdown and rolling Sharpe panels

Usage:
    uv run python analyze.py --split test
    uv run python analyze.py --split train --signal style_momentum
"""
import argparse
import os
import sys

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from config import (
    SPLITS, SIGNALS, MODELS, weights_dir, split_dir,
    z_scores_path, alphas_path,
)
from models import MODEL_REGISTRY


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
    split: str, sig: str, dynamic_models: list[str], out_dir: str
):
    """
    Plot the IC function IC(z) at three checkpoints in time for each model.
    We re-run each model's prediction on a z_grid using the final-state weights
    saved during walk-forward (we re-fit up to each checkpoint date).
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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(dynamic_models)))

    for ax, checkpoint in zip(axes, checkpoints):
        ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="static IC=0.05")
        ax.axhline(0, color="gray", linewidth=0.4)

        for model_name, color in zip(dynamic_models, colors):
            a_path = alphas_path(split, sig, model_name)
            if not os.path.exists(a_path):
                continue

            # Re-fit the model sequentially up to the checkpoint date
            model = MODEL_REGISTRY[model_name]()
            past_z = z_scores.filter(pl.col("date") <= checkpoint)
            past_alphas = pl.read_parquet(a_path).filter(pl.col("date") <= checkpoint)

            joined = past_z.join(
                past_alphas.select(["date", "barrid", "alpha"]),
                on=["date", "barrid"], how="inner",
            ).sort("date", "barrid")

            past_dates = joined["date"].unique().sort()
            for i in range(1, len(past_dates)):
                d = past_dates[i - 1]
                day = joined.filter(pl.col("date") == d)
                z_arr = day["z_score"].to_numpy()
                # alpha_z ≈ IC(z)*z; use as training target (y ≈ alpha/sigma ~ r/sigma)
                y_arr = day["alpha"].to_numpy()
                model.update(z_arr, y_arr)

            alpha_z_pred, _ = model.predict(z_grid)
            with np.errstate(divide="ignore", invalid="ignore"):
                ic_curve = np.where(np.abs(z_grid) > 0.1,
                                    alpha_z_pred / z_grid, np.nan)
            ax.plot(z_grid, ic_curve, label=model_name, color=color)

        ax.set_title(f"{checkpoint}", fontsize=9)
        ax.set_xlabel("Z-score")
        ax.set_xlim(-3, 3)
        ax.yaxis.grid(True)

    axes[0].set_ylabel("Estimated IC(z)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"IC Function — {sig} ({split})", fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"ic_function_{sig}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  IC function → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyse and visualise backtest results")
    parser.add_argument("--split", required=True, choices=list(SPLITS.keys()))
    parser.add_argument("--signal", default=None,
                        help="Single signal (default: all available)")
    args = parser.parse_args()

    split_cfg = SPLITS[args.split]
    signals = [args.signal] if args.signal else SIGNALS
    out_dir = split_dir(args.split)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "backtest_report.txt")
    sys.stdout = Tee(log_path)

    # Load forward returns once
    from signals._asset_signal import ASSETS_PATH
    forward_returns = (
        pl.scan_parquet(ASSETS_PATH)
        .filter(
            (pl.col("date") >= split_cfg["start"]) & (pl.col("date") <= split_cfg["end"]),
            pl.col("in_universe"),
        )
        .select(["date", "barrid", "return"])
        .collect()
        .sort(["barrid", "date"])
        .with_columns(
            pl.col("return").truediv(100).shift(-1).over("barrid").alias("forward_return")
        )
        .select(["date", "barrid", "forward_return"])
    )

    set_academic_style()

    print(f"\nBacktest Summary — {args.split}")
    print(f"{'Strategy':<42} {'Mean%':>8} {'Vol%':>7} {'Sharpe':>8} {'MaxDD%':>8}")
    print("-" * 77)

    all_portfolios: list[pl.DataFrame] = []

    for sig in signals:
        for mod in MODELS:
            pf = load_portfolio_returns(args.split, sig, mod, forward_returns)
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
            all_portfolios.append(pf)

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

        fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                                 gridspec_kw={"height_ratios": [2, 1, 1]})
        ax_cum, ax_dd, ax_sharpe = axes

        for pf in sig_portfolios:
            pf_pd = pf.sort("date").to_pandas().set_index("date")
            rets = pf_pd["return"].fillna(0.0)
            mod_label = pf_pd["name"].iloc[0].split("/")[1]

            cum = np.log1p(rets).cumsum() * 100
            cum_max = cum.cummax()
            dd = cum - cum_max
            rolling_m = rets.rolling(252).mean() * 252 * 100
            rolling_v = rets.rolling(252).std() * np.sqrt(252) * 100
            rolling_s = rolling_m / rolling_v.replace(0, np.nan)

            ax_cum.plot(pf_pd.index, cum, label=mod_label)
            ax_dd.plot(pf_pd.index, dd, label=mod_label, alpha=0.85)
            ax_sharpe.plot(pf_pd.index, rolling_s, label=mod_label, alpha=0.85)

        for ax in axes:
            ax.yaxis.grid(True)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0,
                      fontsize=8)

        ax_cum.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax_cum.set_title(f"Cumulative Return — {sig} ({args.split})", loc="left")
        ax_cum.set_ylabel("Cumulative Log Return (%)")

        ax_dd.set_title("Drawdown", loc="left")
        ax_dd.set_ylabel("Drawdown (%)")

        ax_sharpe.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax_sharpe.set_title("Rolling 1-Year Sharpe", loc="left")
        ax_sharpe.set_ylabel("Sharpe Ratio")

        plt.tight_layout()
        chart_path = os.path.join(out_dir, f"performance_{sig}.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"\nChart → {chart_path}")

    # ── IC function evolution ─────────────────────────────────────────────────
    dynamic_models = [m for m in MODELS if m != "static"]
    for sig in signals:
        visualise_ic_function(args.split, sig, dynamic_models, out_dir)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"\nReport saved to {log_path}")


if __name__ == "__main__":
    main()

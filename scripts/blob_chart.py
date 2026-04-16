"""
Scatter plot of cross-sectional z-score vs. risk-normalized return.

Illustrates the core estimation challenge: a genuine but tiny predictive
relationship (IC ≈ 0.05) buried inside an enormous noise cloud, with
observations concentrated near z = 0 and very few past |z| = 2.

Usage (from the project root):
    uv run python scripts/blob_chart.py

Output:
    paper/blob_chart.png

Data source:
    Tries to load a real z-score parquet from results/recent/z_scores/
    and match it against forward returns from the assets database.
    Falls back to synthetic data (same SNR=0.05 model) if unavailable.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "paper" / "blob_chart.png"
N_SYNTHETIC = 120_000
SNR = 0.05
NOISE_PROPORTION = 1.0 - SNR / (1.0 + SNR)   # ≈ 0.952
SIGNAL_PROPORTION = SNR / (1.0 + SNR)          # ≈ 0.048
N_BINS = 10
MAX_SCATTER = 40_000   # subsample for readability


def _try_load_real() -> tuple[np.ndarray, np.ndarray] | None:
    """Return (z, y) arrays from real parquets, or None if unavailable."""
    try:
        import polars as pl
        from configs import PROJECT_ROOT, SPLIT, SIGNALS

        z_dir = Path(PROJECT_ROOT) / "results" / SPLIT / "z_scores"
        assets_glob = "/home/connerd4/groups/grp_quant/database/research/assets/assets_*.parquet"

        # Pick the first available z-score parquet
        for signal in SIGNALS:
            z_path = z_dir / f"{signal}.parquet"
            if z_path.exists():
                print(f"Loading real data: {signal}")
                z_df = pl.read_parquet(z_path)
                break
        else:
            return None

        # Load forward returns from assets
        assets = (
            pl.scan_parquet(assets_glob)
            .select(["date", "barrid", "return", "specific_risk", "in_universe"])
            .filter(pl.col("in_universe"), pl.col("specific_risk") > 0)
            .collect()
        )

        # Compute forward return: next day's return / today's specific_risk
        assets = (
            assets
            .sort(["barrid", "date"])
            .with_columns(
                pl.col("return").shift(-1).over("barrid").alias("fwd_return")
            )
            .drop_nulls("fwd_return")
        )

        merged = z_df.join(
            assets.select(["date", "barrid", "fwd_return", "specific_risk"]),
            on=["date", "barrid"],
        ).with_columns(
            (pl.col("fwd_return") / (pl.col("specific_risk") / 100)).alias("y")
        )

        z = merged["z_score"].to_numpy()
        y = merged["y"].to_numpy()

        # Drop non-finite
        mask = np.isfinite(z) & np.isfinite(y)
        z, y = z[mask], y[mask]

        # Trim y outliers to ±10 for chart readability (doesn't affect the story)
        mask2 = np.abs(y) < 10
        return z[mask2], y[mask2]

    except Exception as e:
        print(f"Real data unavailable ({e}), using synthetic.")
        return None


def _make_synthetic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    z = np.clip(rng.standard_normal(N_SYNTHETIC), -3.0, 3.0)
    eps = rng.normal(0, np.sqrt(NOISE_PROPORTION), size=N_SYNTHETIC)
    y = SNR * z + eps
    return z, y


def _ols_line(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(z, y, 1)
    return slope, intercept


def _binned_means(
    z: np.ndarray, y: np.ndarray, n_bins: int = N_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(-3.0, 3.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    for k in range(n_bins):
        mask = (z >= edges[k]) & (z < edges[k + 1])
        vals = y[mask]
        if len(vals) > 5:
            means[k] = vals.mean()
            sems[k] = vals.std() / np.sqrt(len(vals))
    return centers, means, sems


def main() -> None:
    result = _try_load_real()
    if result is not None:
        z, y = result
        source_label = "real data"
    else:
        z, y = _make_synthetic()
        source_label = "synthetic (SNR = 0.05)"

    n_total = len(z)
    print(f"Total observations: {n_total:,}  [{source_label}]")

    ic_slope, ic_intercept = _ols_line(z, y)
    print(f"OLS slope (estimated IC): {ic_slope:.4f}")

    bin_centers, bin_means, bin_sems = _binned_means(z, y)

    # Subsample for scatter
    if n_total > MAX_SCATTER:
        idx = np.random.default_rng(0).choice(n_total, MAX_SCATTER, replace=False)
        zs, ys = z[idx], y[idx]
    else:
        zs, ys = z, y

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Scatter cloud
    ax.scatter(
        zs, ys,
        s=1.5, alpha=0.08, color="#555555", rasterized=True,
        label=f"$(z_{{i,t}},\\ y_{{i,t}})$ — {n_total:,} obs.",
        zorder=1,
    )

    # OLS line
    z_line = np.linspace(-3.0, 3.0, 200)
    y_line = ic_slope * z_line + ic_intercept
    ax.plot(
        z_line, y_line,
        color="#c0392b", lw=2.0, zorder=4,
        label=f"OLS: $\\widehat{{IC}} = {ic_slope:.3f}$",
    )

    # Binned means with error bars
    ok = np.isfinite(bin_means)
    ax.errorbar(
        bin_centers[ok], bin_means[ok], yerr=bin_sems[ok],
        fmt="o", color="#2980b9", ms=5, lw=1.4,
        capsize=3, zorder=5,
        label="Bin mean ± SE",
    )

    # Sparse-tail markers
    for xv in [-2.0, 2.0]:
        ax.axvline(xv, color="#888888", lw=0.8, ls="--", zorder=2)
    ax.text(2.05, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.8,
            "sparse\ntails", fontsize=7, color="#888888", va="top")

    ax.axhline(0, color="black", lw=0.5, zorder=2)

    # Annotation
    ax.annotate(
        f"SNR $\\approx$ 0.05\n95\\% noise",
        xy=(0.04, 0.96), xycoords="axes fraction",
        fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
    )

    ax.set_xlabel("Cross-sectional $z$-score", fontsize=10)
    ax.set_ylabel("Risk-normalized return  $y = r / \\sigma$", fontsize=10)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(np.percentile(ys, 0.5), np.percentile(ys, 99.5))
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()

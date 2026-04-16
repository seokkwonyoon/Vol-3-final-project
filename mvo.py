"""
Mean-variance optimization for a single (signal, model) pair (Phase 3).

Reads the alpha parquet produced by train.py and runs backtest_parallel()
to generate per-year weight parquets consumed by analyze.py.

Only processes years whose weight files do not already exist, so
resubmitting after a partial failure picks up exactly where it left off.

Usage:
    uv run python mvo.py --split recent --signal style_momentum --model static
"""
import argparse
import os
import time

import polars as pl
from sf_quant.backtester import backtest_parallel

from configs import (
    SPLITS, SIGNALS, MODELS, GAMMA, CONSTRAINTS,
    alphas_path, weights_dir,
)
from timing import record_elapsed


def _resolve_constraints(names: list[str]):
    from sf_quant.optimizer.constraints import ZeroBeta, ZeroInvestment, LongOnly, UnitBeta
    registry = {
        "ZeroBeta": ZeroBeta,
        "ZeroInvestment": ZeroInvestment,
        "LongOnly": LongOnly,
        "UnitBeta": UnitBeta,
    }
    return [registry[n]() for n in names]


def main():
    parser = argparse.ArgumentParser(description="MVO for one (signal, model) pair")
    parser.add_argument("--split", required=True, choices=list(SPLITS.keys()))
    parser.add_argument("--signal", required=True, choices=SIGNALS)
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()

    a_path = alphas_path(args.split, args.signal, args.model)
    if not os.path.exists(a_path):
        raise FileNotFoundError(
            f"Alpha parquet not found: {a_path}\n"
            "Run train.py (Phase 2) before mvo.py (Phase 3)."
        )

    w_dir = weights_dir(args.split, args.signal, args.model)
    alphas_df = pl.read_parquet(a_path)
    all_years = alphas_df["date"].dt.year().unique().sort().to_list()

    missing_years = [y for y in all_years if not os.path.exists(f"{w_dir}/{y}.parquet")]
    if not missing_years:
        print(f"Weights already exist, skipping: {args.signal}/{args.model}")
        return

    label = f"{args.signal}/{args.model}"
    n_done = len(all_years) - len(missing_years)
    print(f"Running MVO: {label}  "
          f"({len(missing_years)} year(s) remaining, {n_done} already done)",
          flush=True)
    os.makedirs(w_dir, exist_ok=True)

    # Only optimize the years that still need weights
    alphas_missing = alphas_df.filter(pl.col("date").dt.year().is_in(missing_years))

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    constraints = _resolve_constraints(CONSTRAINTS)

    t0 = time.perf_counter()
    weights = backtest_parallel(alphas_missing, constraints, GAMMA, n_cpus)
    elapsed = time.perf_counter() - t0

    for year in missing_years:
        yr_weights = weights.filter(pl.col("date").dt.year() == year)
        yr_weights.write_parquet(f"{w_dir}/{year}.parquet")

    record_elapsed("mvo", f"{args.signal}/{args.model}/{args.split}", elapsed)
    print(f"✓ MVO complete → {w_dir}/  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()

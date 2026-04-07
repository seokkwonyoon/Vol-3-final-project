"""
Walk-forward IC estimation and backtest submission.

For each (signal, model) pair:
  1. Load the cached z-score parquet from compute.py.
  2. Load in-universe asset data (forward returns, specific_risk).
  3. Run the walk-forward IC estimation to produce alpha parquets.
  4. Either submit MVO via Slurm (BacktestRunner) or run it inline
     with backtest_parallel() when --run-mvo is set.

Usage:
    uv run python train.py --split train
    uv run python train.py --split test --signal style_momentum --model kalman_poly
    uv run python train.py --split train --dry-run         # Slurm dry-run
    uv run python train.py --split train --no-backtest     # walk-forward only
    uv run python train.py --split train --run-mvo         # inline MVO (used by run.py)
"""
import argparse
import os
import shutil

import polars as pl
from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig

from config import (
    SPLITS, SIGNALS, MODELS, GAMMA, CONSTRAINTS, BYU_EMAIL,
    z_scores_path, alphas_path, weights_dir, signal_name, split_dir,
    LOOKBACK_DAYS,
)
from pipeline import DynamicICPipeline
from models import MODEL_REGISTRY

# Constraint name → class mapping for --run-mvo inline path
_CONSTRAINT_MAP = {
    "ZeroBeta": None,        # resolved lazily to avoid import at module level
    "ZeroInvestment": None,
}


def _resolve_constraints(names: list[str]):
    """Instantiate sf_quant constraint objects from string names."""
    from sf_quant.optimizer.constraints import ZeroBeta, ZeroInvestment, LongOnly, UnitBeta
    registry = {
        "ZeroBeta": ZeroBeta,
        "ZeroInvestment": ZeroInvestment,
        "LongOnly": LongOnly,
        "UnitBeta": UnitBeta,
    }
    return [registry[n]() for n in names]


def _run_mvo_inline(a_path: str, sig: str, mod: str, split: str, label: str) -> None:
    """
    Run MVO with backtest_parallel() directly (no nested Slurm submission).
    Used by the automated run.py pipeline where each Slurm task handles one pair.
    """
    from sf_quant.backtester import backtest_parallel

    w_dir = weights_dir(split, sig, mod)
    alphas_df = pl.read_parquet(a_path)
    years = alphas_df["date"].dt.year().unique().sort().to_list()

    missing_years = [y for y in years
                     if not os.path.exists(f"{w_dir}/{y}.parquet")]
    if not missing_years:
        print(f"    Weights exist, skipping MVO: {label}")
        return

    print(f"    Running inline MVO: {label}  ({len(missing_years)} year(s))...", flush=True)
    os.makedirs(w_dir, exist_ok=True)

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    constraints = _resolve_constraints(CONSTRAINTS)
    weights = backtest_parallel(alphas_df, constraints, GAMMA, n_cpus)

    for year in years:
        yr_weights = weights.filter(pl.col("date").dt.year() == year)
        yr_weights.write_parquet(f"{w_dir}/{year}.parquet")

    print(f"    ✓ MVO complete → {w_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward train + backtest")
    parser.add_argument("--split", required=True, choices=list(SPLITS.keys()))
    parser.add_argument("--signal", default=None,
                        help="Single signal (default: all signals)")
    parser.add_argument("--model", default=None,
                        help="Single model (default: all models)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print Slurm scripts without submitting")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe old weights before submitting")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Generate alphas only, skip backtest")
    parser.add_argument("--run-mvo", action="store_true",
                        help="Run MVO inline via backtest_parallel() instead of "
                             "submitting a Slurm job (used by run.py)")
    args = parser.parse_args()

    split = SPLITS[args.split]
    signals = [args.signal] if args.signal else SIGNALS
    models = [args.model] if args.model else MODELS
    pipe = DynamicICPipeline()

    slurm_config = SlurmConfig(
        n_cpus=8,
        mem="64G",
        time="00:07:00",
        mail_type="BEGIN,END,FAIL",
        max_concurrent_jobs=30,
    )

    print(f"Split: {args.split} ({split['start']} → {split['end']})")
    print(f"Signals × Models: {len(signals)} × {len(models)} = {len(signals) * len(models)}\n")

    print("Loading asset data...", flush=True)
    assets = pipe.load_assets(split["start"], split["end"])
    print(f"  {assets.height:,} rows\n")

    for sig in signals:
        z_path = z_scores_path(args.split, sig)
        if not os.path.exists(z_path):
            print(f"✗ {sig}: z-score parquet not found at {z_path} — run compute.py first\n")
            continue

        print(f"Loading z-scores: {sig}...", flush=True)
        z_scores = pl.read_parquet(z_path)
        print(f"  {z_scores.height:,} rows")

        for mod in models:
            label = f"{sig} / {mod}"
            a_path = alphas_path(args.split, sig, mod)

            # ── Walk-forward ───────────────────────────────────────────────
            if not os.path.exists(a_path):
                print(f"  Walk-forward: {label}...", flush=True)
                model_instance = MODEL_REGISTRY[mod]()
                alphas = pipe.run_walkforward(
                    z_scores, assets, model_instance, lookback=LOOKBACK_DAYS
                )
                os.makedirs(os.path.dirname(a_path), exist_ok=True)
                alphas.write_parquet(a_path)
                print(f"    ✓ {alphas.height:,} rows → {a_path}")
            else:
                print(f"  Alphas exist, skipping walk-forward: {label}")

            if args.no_backtest:
                continue

            # ── MVO: inline (--run-mvo) or via Slurm (default) ────────────
            if args.run_mvo:
                _run_mvo_inline(a_path, sig, mod, args.split, label)
            else:
                w_dir = weights_dir(args.split, sig, mod)
                if args.clean and os.path.exists(w_dir):
                    shutil.rmtree(w_dir)
                    print(f"    (cleaned {w_dir})")
                os.makedirs(w_dir, exist_ok=True)

                config = BacktestConfig(
                    signal_name=signal_name(sig, mod),
                    data_path=a_path,
                    gamma=GAMMA,
                    project_root=split_dir(args.split),
                    byu_email=BYU_EMAIL,
                    constraints=CONSTRAINTS,
                    slurm=slurm_config,
                )
                runner = BacktestRunner(config)
                runner.submit(dry_run=args.dry_run)
                if not args.dry_run:
                    print(f"    ✓ Submitted backtest: {label}")

        print()

    if args.run_mvo:
        print("Done.")
    elif args.no_backtest:
        print("Done.")
    elif not args.dry_run:
        print("Done. Wait for Slurm jobs to finish before running analyze.py.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()

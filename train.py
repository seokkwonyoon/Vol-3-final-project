"""
Walk-forward IC estimation (Phase 2).

For each (signal, model) pair:
  1. Load the cached z-score parquet from compute.py.
  2. Load in-universe asset data (forward returns, specific_risk).
  3. Run the walk-forward IC estimation to produce an alpha parquet.

The alpha parquet is consumed by mvo.py (Phase 3).

Usage:
    uv run python train.py --split recent
    uv run python train.py --split recent --signal style_momentum --model kalman_poly
"""
import argparse
import os
import time

import polars as pl

from configs import (
    SPLITS, SIGNALS, SELECTED_SIGNALS, MODELS,
    z_scores_path, alphas_path,
    LOOKBACK_DAYS,
)
from pipeline import DynamicICPipeline
from models import MODEL_REGISTRY
from timing import record_elapsed


def main():
    parser = argparse.ArgumentParser(description="Walk-forward IC estimation")
    parser.add_argument("--split", required=True, choices=list(SPLITS.keys()))
    parser.add_argument("--signal", default=None,
                        help="Single signal (default: all signals)")
    parser.add_argument("--model", default=None,
                        help="Single model (default: all models)")
    args = parser.parse_args()

    split = SPLITS[args.split]
    signals = [args.signal] if args.signal else (SELECTED_SIGNALS or SIGNALS)
    models = [args.model] if args.model else MODELS
    pipe = DynamicICPipeline()

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

            if os.path.exists(a_path):
                print(f"  Alphas exist, skipping: {label}")
                continue

            print(f"  Walk-forward: {label}...", flush=True)
            t0 = time.perf_counter()
            model_instance = MODEL_REGISTRY[mod]()
            alphas = pipe.run_walkforward(
                z_scores, assets, model_instance, lookback=LOOKBACK_DAYS
            )
            os.makedirs(os.path.dirname(a_path), exist_ok=True)
            alphas.write_parquet(a_path)
            elapsed = time.perf_counter() - t0
            record_elapsed("train", f"{sig}/{mod}/{args.split}", elapsed)
            print(f"    ✓ {alphas.height:,} rows → {a_path}  ({elapsed:.0f}s)")

        print()

    print("Done.")


if __name__ == "__main__":
    main()

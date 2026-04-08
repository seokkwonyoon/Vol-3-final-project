"""
Pre-compute and cache z-score parquets for all signals.

For each signal, loads the raw signal, cross-sectionally z-scores it
across all stocks on each date, and saves the result to:
    results/{split}/z_scores/{signal}.parquet

Usage:
    uv run python compute.py --split train
    uv run python compute.py --split test --signal style_momentum
"""
import argparse
import os
import time

from config import SPLITS, SIGNALS, z_scores_path
from pipeline import DynamicICPipeline
from timing import record_elapsed


def main():
    parser = argparse.ArgumentParser(description="Pre-compute z-scores for all signals")
    parser.add_argument("--split", required=True, choices=list(SPLITS.keys()))
    parser.add_argument("--signal", default=None,
                        help="Single signal to compute (default: all signals)")
    args = parser.parse_args()

    split = SPLITS[args.split]
    signals = [args.signal] if args.signal else SIGNALS
    pipe = DynamicICPipeline()

    print(f"Split: {args.split} ({split['start']} → {split['end']})")
    print(f"Computing z-scores for {len(signals)} signal(s)...\n")

    for sig in signals:
        out = z_scores_path(args.split, sig)

        if os.path.exists(out):
            print(f"Signal: {sig}  (skipping — {out} already exists)\n")
            continue

        print(f"Signal: {sig}")
        t0 = time.perf_counter()
        try:
            zdf = pipe.compute_zscores(sig, split["start"], split["end"])
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            continue

        os.makedirs(os.path.dirname(out), exist_ok=True)
        zdf.write_parquet(out)
        elapsed = time.perf_counter() - t0
        record_elapsed("compute", f"{sig}/{args.split}", elapsed)
        print(f"  ✓ {zdf.height:,} rows → {out}  ({elapsed:.0f}s)\n")

    print("Done.")


if __name__ == "__main__":
    main()

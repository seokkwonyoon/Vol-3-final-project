"""
Pre-compute and cache z-score parquets for all signals.

For each signal, loads the raw signal, cross-sectionally z-scores it
across all stocks on each date, and saves the result to:
    results/{split}/z_scores/{signal}.parquet

Usage:
    uv run python compute.py
    uv run python compute.py --signal style_momentum
"""
import argparse
import os
import time

from configs import SPLITS, SPLIT, SIGNALS, SELECTED_SIGNALS, z_scores_path
from pipeline import DynamicICPipeline
from timing import record_elapsed, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Pre-compute z-scores for all signals")
    parser.add_argument("--signal", default=None,
                        help="Single signal to compute (default: all signals)")
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    setup_logging("compute", signal=args.signal or "all")

    split = SPLITS[SPLIT]
    signals = [args.signal] if args.signal else (SELECTED_SIGNALS or SIGNALS)
    pipe = DynamicICPipeline()

    print(f"Split: {SPLIT} ({split['start']} → {split['end']})")
    print(f"Computing z-scores for {len(signals)} signal(s)...\n")

    for sig in signals:
        out = z_scores_path(SPLIT, sig)

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
        record_elapsed("compute", f"{sig}/{SPLIT}", elapsed)
        print(f"  ✓ {zdf.height:,} rows → {out}  ({elapsed:.0f}s)\n")

    print("Done.")


if __name__ == "__main__":
    main()

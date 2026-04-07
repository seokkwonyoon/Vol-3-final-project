"""
Single-command fire-and-forget pipeline orchestrator.

Generates and submits a 3-phase Slurm job chain:

  Phase 1  (job array, 1 task/signal)          compute.py  --signal $SIG
  Phase 2  (job array, 1 task/signal×model)    train.py    --signal $SIG --model $MOD --run-mvo
  Phase 3  (single job)                        analyze.py  --split $SPLIT

Each phase waits for ALL tasks of the previous phase to succeed before
starting (Slurm --dependency=afterok on an array job).  You receive a
failure email if any task fails, and a completion email when Phase 3 ends.

Usage:
    uv run python run.py --split train
    uv run python run.py --split test --dry-run          # print scripts, no submission
    uv run python run.py --split train --signal style_momentum  # single signal
    uv run python run.py --split train --model static           # single model

Resource estimates per phase:
    Phase 1: 4 CPUs, 32G, 30 min  (z-score computation per signal)
    Phase 2: 16 CPUs, 64G, 3 hr   (walk-forward + inline MVO per pair)
    Phase 3: 4 CPUs, 32G, 30 min  (chart generation)
"""
import argparse
import os
import subprocess
import sys
import tempfile
import textwrap

from config import (
    SIGNALS, MODELS, SPLITS, BYU_EMAIL, PROJECT_ROOT,
    SLURM_TIME_COMPUTE, SLURM_TIME_TRAIN, SLURM_TIME_ANALYZE,
)


# ── Slurm submission helper ───────────────────────────────────────────────────

def sbatch_submit(script: str, dry_run: bool) -> str:
    """
    Write *script* to a temp file, submit with ``sbatch --parsable``,
    and return the numeric job ID as a string.

    In dry-run mode, prints the script and returns the string "DRY_RUN".
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        if dry_run:
            print(script)
            print("─" * 60)
            return "DRY_RUN"
        result = subprocess.run(
            ["sbatch", "--parsable", path],
            capture_output=True, text=True, check=True,
        )
        # --parsable output: "JOBID" or "JOBID;cluster"
        return result.stdout.strip().split(";")[0]
    except subprocess.CalledProcessError as e:
        print(f"sbatch failed:\n{e.stderr}", file=sys.stderr)
        raise
    finally:
        os.unlink(path)


# ── Script generators ─────────────────────────────────────────────────────────

def _header(
    job_name: str,
    log_prefix: str,
    n_cpus: int,
    mem: str,
    time: str,
    array: str | None = None,
    dependency: str | None = None,
    mail_type: str = "FAIL",
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={PROJECT_ROOT}/logs/{log_prefix}_%A_%a.out",
        f"#SBATCH --error={PROJECT_ROOT}/logs/{log_prefix}_%A_%a.err",
    ]
    if array:
        lines.append(f"#SBATCH --array={array}")
    if dependency:
        lines.append(f"#SBATCH --dependency=afterok:{dependency}")
    lines += [
        f"#SBATCH --cpus-per-task={n_cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --mail-user={BYU_EMAIL}",
        f"#SBATCH --mail-type={mail_type}",
        "",
        f"source {PROJECT_ROOT}/.venv/bin/activate",
        f"cd {PROJECT_ROOT}",
        "",
    ]
    return "\n".join(lines)


def phase1_script(split: str, signals: list[str]) -> str:
    sigs_bash = " ".join(f'"{s}"' for s in signals)
    header = _header(
        job_name=f"dic_compute_{split}",
        log_prefix="compute",
        n_cpus=4, mem="32G", time=SLURM_TIME_COMPUTE,
        array=f"0-{len(signals) - 1}",
        mail_type="FAIL",
    )
    body = textwrap.dedent(f"""\
        SIGNALS=({sigs_bash})
        SIG="${{SIGNALS[$SLURM_ARRAY_TASK_ID]}}"

        echo "Phase 1 | split={split} | signal=$SIG"
        uv run python compute.py --split {split} --signal "$SIG"
    """)
    return header + body


def phase2_script(
    split: str,
    signals: list[str],
    models: list[str],
    dep_id: str,
) -> str:
    pairs = [(s, m) for s in signals for m in models]
    sigs_bash = " ".join(f'"{s}"' for s, _ in pairs)
    mods_bash = " ".join(f'"{m}"' for _, m in pairs)
    header = _header(
        job_name=f"dic_train_{split}",
        log_prefix="train",
        n_cpus=16, mem="64G", time=SLURM_TIME_TRAIN,
        array=f"0-{len(pairs) - 1}",
        dependency=dep_id,
        mail_type="FAIL",
    )
    body = textwrap.dedent(f"""\
        SIGNALS=({sigs_bash})
        MODELS=({mods_bash})
        SIG="${{SIGNALS[$SLURM_ARRAY_TASK_ID]}}"
        MOD="${{MODELS[$SLURM_ARRAY_TASK_ID]}}"

        echo "Phase 2 | split={split} | signal=$SIG | model=$MOD"
        uv run python train.py --split {split} --signal "$SIG" --model "$MOD" --run-mvo
    """)
    return header + body


def phase3_script(split: str, dep_id: str) -> str:
    header = _header(
        job_name=f"dic_analyze_{split}",
        log_prefix="analyze",
        n_cpus=4, mem="32G", time=SLURM_TIME_ANALYZE,
        dependency=dep_id,
        mail_type="END,FAIL",
    )
    body = textwrap.dedent(f"""\
        echo "Phase 3 | split={split} | analysis"
        uv run python analyze.py --split {split}
    """)
    return header + body


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Submit the full dynamic IC pipeline to Slurm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              uv run python run.py --split train
              uv run python run.py --split test --dry-run
              uv run python run.py --split train --signal style_momentum --model kalman_poly
        """),
    )
    parser.add_argument("--split", required=True, choices=list(SPLITS.keys()))
    parser.add_argument("--signal", default=None,
                        help="Restrict to a single signal (default: all)")
    parser.add_argument("--model", default=None,
                        help="Restrict to a single model (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated scripts without submitting")
    args = parser.parse_args()

    signals = [args.signal] if args.signal else SIGNALS
    models = [args.model] if args.model else MODELS
    n_pairs = len(signals) * len(models)

    os.makedirs(f"{PROJECT_ROOT}/logs", exist_ok=True)

    print(f"Split:   {args.split}")
    print(f"Signals: {len(signals)}   Models: {len(models)}   Pairs: {n_pairs}")
    if args.dry_run:
        print("(dry-run — scripts will be printed, nothing submitted)\n")

    # Phase 1: compute z-scores (one task per signal)
    s1 = phase1_script(args.split, signals)
    jid1 = sbatch_submit(s1, args.dry_run)
    print(f"Phase 1 submitted: job {jid1}  ({len(signals)}-task array)")

    # Phase 2: walk-forward + MVO (one task per signal×model pair)
    s2 = phase2_script(args.split, signals, models, dep_id=jid1)
    jid2 = sbatch_submit(s2, args.dry_run)
    print(f"Phase 2 submitted: job {jid2}  ({n_pairs}-task array)  [afterok:{jid1}]")

    # Phase 3: analysis (single job, depends on all of Phase 2)
    s3 = phase3_script(args.split, dep_id=jid2)
    jid3 = sbatch_submit(s3, args.dry_run)
    print(f"Phase 3 submitted: job {jid3}  (single job)            [afterok:{jid2}]")

    if not args.dry_run:
        print(f"\nPipeline queued.  Email → {BYU_EMAIL} on failure or completion.")
        print(f"Monitor with:  squeue -u $USER")


if __name__ == "__main__":
    main()

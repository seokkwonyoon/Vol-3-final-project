"""
Single-command fire-and-forget pipeline orchestrator.

Generates and submits a 4-phase Slurm job chain:

  Phase 1  (job array, 1 task/signal)          compute.py  --signal $SIG
  Phase 2  (job array, 1 task/signal×model)    train.py    --signal $SIG --model $MOD
  Phase 3  (job array, 1 task/signal×model)    mvo.py      --signal $SIG --model $MOD
  Phase 4  (single job)                        analyze.py

Dependency structure:
  Phase 2  afterok:Phase1      (all z-scores must exist before any walk-forward)
  Phase 3  aftercorr:Phase2    (mvo task i starts as soon as train task i succeeds —
                                 fast pairs don't wait for slow ones like nadaraya_watson)
  Phase 4  afterok:Phase3      (all weights must exist before analysis)

Time limits are set dynamically from logs/timing.json when history exists,
falling back to config defaults on the first run.  Run timing.py to inspect
the current estimates.

Split and signal selection are read from configs/default.py (SPLIT, SELECTED_SIGNALS).
Pass --config to override with a different config file.

Usage:
    uv run python submit.py
    uv run python submit.py --dry-run                   # print scripts, no submission
    uv run python submit.py --config configs/test.py    # use a custom config
    uv run python submit.py --model static
    uv run python submit.py --clear                     # wipe results for active split, then rerun
    uv run python submit.py --clear --dry-run           # show what would be deleted, no action
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

from configs import (
    SIGNALS, SELECTED_SIGNALS, MODELS, SPLITS, SPLIT, BYU_EMAIL, PROJECT_ROOT,
    SLURM_TIME_COMPUTE, SLURM_CPUS_COMPUTE, SLURM_MEM_COMPUTE,
    SLURM_TIME_TRAIN,   SLURM_CPUS_TRAIN,   SLURM_MEM_TRAIN,
    SLURM_TIME_MVO,     SLURM_CPUS_MVO,     SLURM_MEM_MVO,
    SLURM_TIME_ANALYZE, SLURM_CPUS_ANALYZE, SLURM_MEM_ANALYZE,
)
from timing import estimate_seconds, to_slurm_time


# ── Result clearing ──────────────────────────────────────────────────────────

def clear_split(split: str, dry_run: bool) -> None:
    """Delete all computed results for *split* so the pipeline starts completely fresh."""
    target = os.path.join(PROJECT_ROOT, "results", split)

    if not os.path.exists(target):
        print(f"Nothing to clear: {target} does not exist.")
        return

    # Summarise what will be removed
    subdirs = sorted(
        d for d in os.listdir(target)
        if os.path.isdir(os.path.join(target, d))
    )
    files = sorted(
        f for f in os.listdir(target)
        if os.path.isfile(os.path.join(target, f))
    )
    print(f"\nWill delete: {target}")
    for name in subdirs:
        print(f"  {name}/")
    for name in files:
        print(f"  {name}")

    if dry_run:
        print("(dry-run — nothing deleted)\n")
        return

    confirm = input(f"\nDelete all results for split '{split}'? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    shutil.rmtree(target)
    print(f"Cleared {target}\n")


# ── Slurm submission helper ───────────────────────────────────────────────────

def sbatch_submit(script: str, dry_run: bool) -> str:
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
        return result.stdout.strip().split(";")[0]
    except subprocess.CalledProcessError as e:
        print(f"sbatch failed:\n{e.stderr}", file=sys.stderr)
        raise
    finally:
        os.unlink(path)


# ── Script header generator ───────────────────────────────────────────────────

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
        lines.append(f"#SBATCH --dependency={dependency}")
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


# ── Time limit helpers ────────────────────────────────────────────────────────

def _hms_to_sec(hms: str) -> float:
    """HH:MM:SS → seconds."""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _phase1_time(signals: list[str], split: str) -> str:
    default = _hms_to_sec(SLURM_TIME_COMPUTE)
    worst = max(
        estimate_seconds("compute", f"{sig}/{split}", default)
        for sig in signals
    )
    return to_slurm_time(worst)


def _phase2_time(pairs: list[tuple[str, str]], split: str) -> str:
    default = _hms_to_sec(SLURM_TIME_TRAIN)
    worst = max(
        estimate_seconds("train", f"{sig}/{mod}/{split}", default)
        for sig, mod in pairs
    )
    return to_slurm_time(worst)


def _phase3_time(pairs: list[tuple[str, str]], split: str) -> str:
    default = _hms_to_sec(SLURM_TIME_MVO)
    worst = max(
        estimate_seconds("mvo", f"{sig}/{mod}/{split}", default)
        for sig, mod in pairs
    )
    return to_slurm_time(worst)


def _phase4_time(split: str) -> str:
    default = _hms_to_sec(SLURM_TIME_ANALYZE)
    return to_slurm_time(estimate_seconds("analyze", split, default))


# ── Phase script generators ───────────────────────────────────────────────────

def phase1_script(split: str, signals: list[str], time_limit: str,
                  config: str | None = None) -> str:
    sigs_bash = " ".join(f'"{s}"' for s in signals)
    cfg = f" --config {config}" if config else ""
    header = _header(
        job_name=f"dic_compute_{split}",
        log_prefix="compute",
        n_cpus=SLURM_CPUS_COMPUTE, mem=SLURM_MEM_COMPUTE, time=time_limit,
        array=f"0-{len(signals) - 1}",
        mail_type="FAIL",
    )
    body = textwrap.dedent(f"""\
        SIGNALS=({sigs_bash})
        SIG="${{SIGNALS[$SLURM_ARRAY_TASK_ID]}}"

        echo "Phase 1 | split={split} | signal=$SIG"
        uv run python compute.py --signal "$SIG"{cfg}
    """)
    return header + body


def phase2_script(split: str, pairs: list[tuple[str, str]], dep_id: str,
                  time_limit: str, config: str | None = None) -> str:
    sigs_bash = " ".join(f'"{s}"' for s, _ in pairs)
    mods_bash = " ".join(f'"{m}"' for _, m in pairs)
    cfg = f" --config {config}" if config else ""
    header = _header(
        job_name=f"dic_train_{split}",
        log_prefix="train",
        n_cpus=SLURM_CPUS_TRAIN, mem=SLURM_MEM_TRAIN, time=time_limit,
        array=f"0-{len(pairs) - 1}",
        dependency=f"afterok:{dep_id}",
        mail_type="FAIL",
    )
    body = textwrap.dedent(f"""\
        SIGNALS=({sigs_bash})
        MODELS=({mods_bash})
        SIG="${{SIGNALS[$SLURM_ARRAY_TASK_ID]}}"
        MOD="${{MODELS[$SLURM_ARRAY_TASK_ID]}}"

        echo "Phase 2 | split={split} | signal=$SIG | model=$MOD"
        uv run python train.py --signal "$SIG" --model "$MOD"{cfg}
    """)
    return header + body


def phase3_script(split: str, pairs: list[tuple[str, str]], dep_id: str,
                  time_limit: str, config: str | None = None) -> str:
    sigs_bash = " ".join(f'"{s}"' for s, _ in pairs)
    mods_bash = " ".join(f'"{m}"' for _, m in pairs)
    cfg = f" --config {config}" if config else ""
    header = _header(
        job_name=f"dic_mvo_{split}",
        log_prefix="mvo",
        n_cpus=SLURM_CPUS_MVO, mem=SLURM_MEM_MVO, time=time_limit,
        array=f"0-{len(pairs) - 1}",
        # aftercorr: mvo task i starts as soon as train task i succeeds.
        # Fast pairs (static, kalman_poly) don't wait for slow ones (nadaraya_watson).
        dependency=f"aftercorr:{dep_id}",
        mail_type="FAIL",
    )
    body = textwrap.dedent(f"""\
        SIGNALS=({sigs_bash})
        MODELS=({mods_bash})
        SIG="${{SIGNALS[$SLURM_ARRAY_TASK_ID]}}"
        MOD="${{MODELS[$SLURM_ARRAY_TASK_ID]}}"

        echo "Phase 3 | split={split} | signal=$SIG | model=$MOD"
        uv run python mvo.py --signal "$SIG" --model "$MOD"{cfg}
    """)
    return header + body


def phase4_script(split: str, dep_id: str, time_limit: str,
                  config: str | None = None) -> str:
    cfg = f" --config {config}" if config else ""
    header = _header(
        job_name=f"dic_analyze_{split}",
        log_prefix="analyze",
        n_cpus=SLURM_CPUS_ANALYZE, mem=SLURM_MEM_ANALYZE, time=time_limit,
        dependency=f"afterany:{dep_id}",
        mail_type="END,FAIL",
    )
    body = textwrap.dedent(f"""\
        echo "Phase 4 | split={split} | analysis"
        uv run python analyze.py{cfg}
    """)
    return header + body


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Submit the full dynamic IC pipeline to Slurm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              uv run python submit.py
              uv run python submit.py --dry-run
              uv run python submit.py --config configs/test.py
              uv run python submit.py --model static
              uv run python submit.py --clear             (wipe split results, then rerun)
              uv run python submit.py --clear --dry-run   (preview what would be deleted)

            Split and signals are configured in configs/default.py (SPLIT, SELECTED_SIGNALS).
            Pass --config to override with a different config file.

            To inspect estimated time limits from past runs:
              uv run python timing.py
        """),
    )
    parser.add_argument("--config", default=None,
                        help="Config file to use (default: configs/default.py)")
    parser.add_argument("--model", default=None,
                        help="Restrict to a single model (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated scripts without submitting")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing results for the active split before submitting "
                             "(forces a clean rerun; prompts for confirmation)")
    args = parser.parse_args()

    signals = SELECTED_SIGNALS or SIGNALS
    models  = [args.model] if args.model else MODELS
    pairs   = [(s, m) for s in signals for m in models]
    split   = SPLIT

    if args.clear:
        clear_split(split, args.dry_run)

    os.makedirs(f"{PROJECT_ROOT}/logs", exist_ok=True)

    # Compute dynamic time limits (falls back to config defaults if no history)
    t1 = _phase1_time(signals, split)
    t2 = _phase2_time(pairs, split)
    t3 = _phase3_time(pairs, split)
    t4 = _phase4_time(split)

    print(f"Split:   {split}")
    print(f"Signals: {len(signals)}   Models: {len(models)}   Pairs: {len(pairs)}")
    print(f"Time limits:  Phase1={t1}  Phase2={t2}  Phase3={t3}  Phase4={t4}")
    if args.dry_run:
        print("(dry-run — scripts will be printed, nothing submitted)\n")

    cfg = args.config

    s1  = phase1_script(split, signals, t1, cfg)
    jid1 = sbatch_submit(s1, args.dry_run)
    print(f"Phase 1 submitted: job {jid1}  ({len(signals)}-task array)  compute   [{t1}]")

    s2  = phase2_script(split, pairs, jid1, t2, cfg)
    jid2 = sbatch_submit(s2, args.dry_run)
    print(f"Phase 2 submitted: job {jid2}  ({len(pairs)}-task array)   train     [{t2}]  [afterok:{jid1}]")

    s3  = phase3_script(split, pairs, jid2, t3, cfg)
    jid3 = sbatch_submit(s3, args.dry_run)
    print(f"Phase 3 submitted: job {jid3}  ({len(pairs)}-task array)   mvo       [{t3}]  [aftercorr:{jid2}]")

    s4  = phase4_script(split, jid3, t4, cfg)
    jid4 = sbatch_submit(s4, args.dry_run)
    print(f"Phase 4 submitted: job {jid4}  (single job)               analyze   [{t4}]  [afterok:{jid3}]")

    if not args.dry_run:
        print(f"\nPipeline queued.  Email → {BYU_EMAIL} on failure or completion.")
        print(f"Monitor with:  squeue -u $USER")


if __name__ == "__main__":
    main()

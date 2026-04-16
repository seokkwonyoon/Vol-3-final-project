"""
Wall-clock timing tracker for Slurm job time limit estimation.

Each script calls record_elapsed() at the end of a successful run.
submit.py calls estimate_seconds() to set dynamic --time limits.

Storage: logs/timing.json
  {
    "compute": { "style_momentum/recent": [118.5, 121.3, ...] },
    "train":   { "style_momentum/static/recent": [45.3, ...] },
    "mvo":     { "style_momentum/static/recent": [4500.2, ...] },
    "analyze": { "recent": [280.5, ...] }
  }

Each list stores the last MAX_HISTORY observed wall-clock seconds.
Estimates use max(observed) × BUFFER, with a minimum floor of the
config default so the first run is always safe.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_TIMING_PATH = Path(__file__).parent / "logs" / "timing.json"
_LOGS_DIR    = Path(__file__).parent / "logs"


# ── Run logging ───────────────────────────────────────────────────────────────

def setup_logging(phase: str, **tags: str) -> Path:
    """
    Tee stdout and stderr to a timestamped log file for this phase.

    Call once at the top of main() in each pipeline script.  All print()
    output and Python tracebacks will appear on the terminal *and* in the
    log file, so crashes are captured without any extra steps.

    Parameters
    ----------
    phase : str
        One of "compute", "train", "mvo", "analyze", "submit".
    **tags : str
        Optional key-value pairs added to the filename, e.g.
        signal="style_momentum", model="kalman_poly".

    Returns
    -------
    Path
        Path to the log file that was opened.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts] + [str(v) for v in tags.values() if v is not None]
    log_dir = _LOGS_DIR / phase
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("_".join(parts) + ".log")

    _f = open(log_path, "w", buffering=1)

    class _Tee:
        def __init__(self, stream):
            self._stream = stream
        def write(self, msg):
            self._stream.write(msg)
            _f.write(msg)
            return len(msg)
        def flush(self):
            self._stream.flush()
            _f.flush()
        def __getattr__(self, name):
            return getattr(self._stream, name)

    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)

    print(f"[log] {log_path}")
    return log_path
_MAX_HISTORY = 10     # observations kept per key
_BUFFER      = 2.0    # multiply observed max by this factor


# ── Public API ────────────────────────────────────────────────────────────────

def record_elapsed(phase: str, key: str, elapsed_sec: float) -> None:
    """
    Append *elapsed_sec* to the history for (*phase*, *key*).

    Parameters
    ----------
    phase : str
        One of "compute", "train", "mvo", "analyze".
    key : str
        Identifies the specific task, e.g. "style_momentum/static/recent".
    elapsed_sec : float
        Wall-clock seconds for the completed run.
    """
    data = _load()
    bucket = data.setdefault(phase, {}).setdefault(key, [])
    bucket.append(round(elapsed_sec, 1))
    data[phase][key] = bucket[-_MAX_HISTORY:]
    _atomic_save(data)


def estimate_seconds(phase: str, key: str, default_sec: float) -> float:
    """
    Return estimated walltime in seconds for (*phase*, *key*).

    Uses max(observed) × BUFFER, or *default_sec* when no history exists.
    The result is always at least *default_sec* (never shrinks below floor).
    """
    data = _load()
    obs = data.get(phase, {}).get(key, [])
    if not obs:
        return default_sec
    return max(max(obs) * _BUFFER, default_sec)


def to_slurm_time(sec: float) -> str:
    """Convert seconds to HH:MM:SS Slurm format."""
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def show_summary() -> None:
    """Print a human-readable timing summary to stdout."""
    data = _load()
    if not data:
        print("No timing data recorded yet.")
        return
    for phase, keys in sorted(data.items()):
        print(f"\n{phase}:")
        for key, obs in sorted(keys.items()):
            if not obs:
                continue
            avg = sum(obs) / len(obs)
            mx  = max(obs)
            est = to_slurm_time(mx * _BUFFER)
            print(f"  {key:<50}  n={len(obs):2d}  "
                  f"avg={avg/60:6.1f}m  max={mx/60:6.1f}m  "
                  f"→ limit {est}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load() -> dict:
    if _TIMING_PATH.exists():
        try:
            return json.loads(_TIMING_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _atomic_save(data: dict) -> None:
    _TIMING_PATH.parent.mkdir(exist_ok=True)
    tmp = _TIMING_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, _TIMING_PATH)


if __name__ == "__main__":
    show_summary()

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
from pathlib import Path

_TIMING_PATH = Path(__file__).parent / "logs" / "timing.json"
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

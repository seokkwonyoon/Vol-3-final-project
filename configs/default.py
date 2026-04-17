"""
Shared configuration for the dynamic IC pipeline.
"""
import datetime as dt
from pathlib import Path

# ── Date splits ─────────────────────────────────────────────────────────────
SPLITS = {
    "full":   {"start": dt.date(1995, 1, 1), "end": dt.date(2025, 1, 1)},
    "recent":  {"start": dt.date(2010, 1, 1), "end": dt.date(2025, 1, 1)},
    "recent2": {"start": dt.date(2020, 1, 1), "end": dt.date(2025, 1, 1)},
    "test": {"start": dt.date(2022, 1, 1), "end": dt.date(2024, 1, 1)}
}

# Active split — change this to switch between "full", "recent", or "test"
SPLIT = "full"

# ── Strategy parameters ──────────────────────────────────────────────────────
STATIC_IC = 0.05             # Baseline IC for the static benchmark
GAMMA = 200                  # Risk-aversion for MVO
LOOKBACK_DAYS = 504          # Rolling window fed to IC models (trading days)

# ── Signal-to-Noise Ratio (SNR) ──────────────────────────────────────────────
# Used to tune ML models for low-SNR environments.
# Signal Proportion = SNR / (1 + SNR)  =>  1 - Signal Proportion = Noise Proportion
SNR = 0.05                   # 5% signal, 95% noise
NOISE_PROPORTION = 1.0 - (SNR / (1.0 + SNR))  # ~0.95 for SNR=0.05
SIGNAL_PROPORTION = SNR / (1.0 + SNR)          # ~0.05 for SNR=0.05
NOISE_TO_SIGNAL_RATIO = NOISE_PROPORTION / SIGNAL_PROPORTION  # ~19.0

# Window parameters shared with momentum signals
MOMENTUM_WINDOW = 231        # 11-to-1 month lookback
MOMENTUM_SKIP = 21           # Skip most recent month
REVERSAL_WINDOW = 21         # Short-term reversal window

# Minimum price filter for universe
MIN_PRICE = 5.0

# ── Barra factor groupings ───────────────────────────────────────────────────
STYLE_FACTORS = [
    "BETA", "DIVYILD", "EARNQLTY", "EARNYILD", "GROWTH", "LEVERAGE",
    "LIQUIDTY", "LTREVRSL", "MGMTQLTY", "MIDCAP", "MOMENTUM", "PROFIT",
    "PROSPECT", "RESVOL", "SIZE", "VALUE",
]

INDUSTRY_FACTORS = [
    "AERODEF", "AIRLINES", "ALUMSTEL", "APPAREL", "AUTO", "BANKS",
    "BEVTOB", "BIOLIFE", "BLDGPROD", "CHEM", "CNSTENG", "CNSTMACH",
    "CNSTMATL", "COMMEQP", "COMPELEC", "COMSVCS", "CONGLOM", "CONTAINR",
    "DISTRIB", "DIVFIN", "ELECEQP", "ELECUTIL", "FOODPROD", "FOODRET",
    "GASUTIL", "HLTHEQP", "HLTHSVCS", "HOMEBLDG", "HOUSEDUR", "INDMACH",
    "INSURNCE", "INTERNET", "LEISPROD", "LEISSVCS", "LIFEINS", "MEDIA",
    "MGDHLTH", "MULTUTIL", "OILGSCON", "OILGSDRL", "OILGSEQP", "OILGSEXP",
    "PAPER", "PHARMA", "PRECMTLS", "PSNLPROD", "REALEST", "RESTAUR",
    "ROADRAIL", "SEMICOND", "SEMIEQP", "SOFTWARE", "SPLTYRET", "SPTYCHEM",
    "SPTYSTOR", "TELECOM", "TRADECO", "TRANSPRT", "WIRELESS",
]

# ── Available signals ────────────────────────────────────────────────────────
SIGNALS = [
    "style_momentum",
    "industry_momentum",
    "idiosyncratic_momentum",
    "style_reversal",
    "industry_reversal",
    "idiosyncratic_reversal",
    "idiosyncratic_volatility",
    "betting_against_beta",
]

# Subset of signals to train on (set to None to use all signals)
SELECTED_SIGNALS = None  # or e.g., ["style_momentum", "industry_momentum"]

# ── Available IC models ──────────────────────────────────────────────────────
MODELS = [
    "static",
    "kalman_poly",
    "rbf_rls",
    "binned_kalman",
    "nadaraya_watson",
    "gaussian_process_regression",
    "kernel_ridge_regression",
]

# ── Backtest constraints ─────────────────────────────────────────────────────
CONSTRAINTS = ["ZeroBeta", "ZeroInvestment"]

# ── Slurm resource specs ──────────────────────────────────────────────────────
# Phase 1: z-score computation (one task per signal)
SLURM_TIME_COMPUTE = "00:30:00"
SLURM_CPUS_COMPUTE = 4
SLURM_MEM_COMPUTE  = "32G"

# Phase 2: walk-forward only (one task per signal×model)
# nadaraya_watson is the bottleneck — give it plenty of headroom
SLURM_TIME_TRAIN   = "12:00:00"
SLURM_CPUS_TRAIN   = 4     # sequential loop; extra CPUs don't help
SLURM_MEM_TRAIN    = "32G"

# Phase 3: MVO via Ray (one task per signal×model)
SLURM_TIME_MVO     = "07:00:00"   # 15 yr × 5 min/yr = 75 min + buffer
SLURM_CPUS_MVO     = 16           # Ray fans out across all CPUs
SLURM_MEM_MVO      = "64G"

# Phase 4: analysis and chart generation
SLURM_TIME_ANALYZE = "02:00:00"
SLURM_CPUS_ANALYZE = 4
SLURM_MEM_ANALYZE  = "32G"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
FACTORS_PATH = "/home/connerd4/groups/grp_quant/database/research/factors/factors_*.parquet"
EXPOSURES_PATH = "/home/connerd4/groups/grp_quant/database/research/exposures/exposures_*.parquet"
BYU_EMAIL = "connerd4@byu.edu"


# ── Path helpers ─────────────────────────────────────────────────────────────
def split_dir(split: str) -> str:
    return f"{PROJECT_ROOT}/results/{split}"


def z_scores_path(split: str, signal: str) -> str:
    """Cached z-score parquet for a signal."""
    return f"{split_dir(split)}/z_scores/{signal}.parquet"


def alphas_path(split: str, signal: str, model: str) -> str:
    """Alpha parquet produced by walk-forward for (signal, model) pair."""
    return f"{split_dir(split)}/alphas/{signal}/{model}.parquet"


def weights_dir(split: str, signal: str, model: str) -> str:
    """Weights directory consumed by sf_backtester."""
    return f"{split_dir(split)}/weights/{signal}/{model}/{GAMMA}"


def signal_name(signal: str, model: str) -> str:
    """Human-readable name used as the sf_backtester signal_name."""
    return f"dynamic_ic_{signal}_{model}"

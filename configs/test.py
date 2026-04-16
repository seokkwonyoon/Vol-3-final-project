"""
Lean test configuration for smoke testing the pipeline.

This config uses a tiny split, a single signal, and a small set of models
so that compute, training, MVO, and analysis can run quickly.
"""
from configs.default import *

SPLITS = {
    "smoke": {"start": dt.date(2023, 1, 2), "end": dt.date(2023, 1, 31)},
}

SIGNALS = ["style_momentum", "industry_momentum"]
SELECTED_SIGNALS = SIGNALS
MODELS = [
    "static",
    "gaussian_process_regression",
    "kernel_ridge_regression",
]
LOOKBACK_DAYS = 20

SLURM_TIME_COMPUTE = "00:10:00"
SLURM_CPUS_COMPUTE = 1
SLURM_MEM_COMPUTE = "8G"

SLURM_TIME_TRAIN = "00:15:00"
SLURM_CPUS_TRAIN = 1
SLURM_MEM_TRAIN = "8G"

SLURM_TIME_MVO = "00:30:00"
SLURM_CPUS_MVO = 2
SLURM_MEM_MVO = "8G"

SLURM_TIME_ANALYZE = "00:05:00"
SLURM_CPUS_ANALYZE = 1
SLURM_MEM_ANALYZE = "4G"

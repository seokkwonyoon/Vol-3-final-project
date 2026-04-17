"""
Configuration that uses only the Gaussian Process and Kernel Ridge models.
"""
from configs.default import *

MODELS = [
    "gaussian_process_regression",
    "kernel_ridge_regression",
    "kalman_poly",
    "static",
]

# no nadaraya-watson so training is quite fast
SLURM_TIME_TRAIN   = "1:00:00"

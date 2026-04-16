"""
Configuration that uses only the Gaussian Process and Kernel Ridge models.
"""
from configs.default import *

MODELS = [
    "gaussian_process_regression",
    "kernel_ridge_regression",
]

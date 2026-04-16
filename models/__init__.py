"""
Registry of available IC models.

Usage
-----
from models import MODEL_REGISTRY
model = MODEL_REGISTRY["kalman_poly"]()
"""
from models.static import StaticIC
from models.kalman_poly import KalmanPolyIC
from models.rbf_rls import RbfRlsIC
from models.binned_kalman import BinnedKalmanIC
from models.nadaraya_watson import NadarayaWatsonIC
from models.gaussian_process_regression import GaussianProcessRegressionIC
from models.kernel_ridge_regression import KernelRidgeRegressionIC

MODEL_REGISTRY: dict[str, type] = {
    "static":                       StaticIC,
    "kalman_poly":                  KalmanPolyIC,
    "rbf_rls":                      RbfRlsIC,
    "binned_kalman":                BinnedKalmanIC,
    "nadaraya_watson":              NadarayaWatsonIC,
    "gaussian_process_regression":  GaussianProcessRegressionIC,
    "kernel_ridge_regression":      KernelRidgeRegressionIC,
}

__all__ = [
    "MODEL_REGISTRY",
    "StaticIC",
    "KalmanPolyIC",
    "RbfRlsIC",
    "BinnedKalmanIC",
    "NadarayaWatsonIC",
    "GaussianProcessRegressionIC",
    "KernelRidgeRegressionIC",
]

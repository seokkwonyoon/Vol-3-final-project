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

MODEL_REGISTRY: dict[str, type] = {
    "static":           StaticIC,
    "kalman_poly":      KalmanPolyIC,
    "rbf_rls":          RbfRlsIC,
    "binned_kalman":    BinnedKalmanIC,
    "nadaraya_watson":  NadarayaWatsonIC,
}

__all__ = [
    "MODEL_REGISTRY",
    "StaticIC",
    "KalmanPolyIC",
    "RbfRlsIC",
    "BinnedKalmanIC",
    "NadarayaWatsonIC",
]

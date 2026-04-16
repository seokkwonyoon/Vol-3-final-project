"""
Static IC baseline: IC(z) = STATIC_IC for all z.

alpha_z = STATIC_IC * z
"""
import numpy as np
from models.base import ICModel
from configs import STATIC_IC


class StaticIC(ICModel):
    """Fixed IC = STATIC_IC. No learning, no update."""

    def update(self, z: np.ndarray, y: np.ndarray) -> None:
        pass  # stateless

    def predict(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha_z = STATIC_IC * z
        variance_z = np.zeros_like(z)
        return alpha_z, variance_z

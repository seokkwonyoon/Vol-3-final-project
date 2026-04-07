"""
Abstract base class for all IC function estimators.

Convention
----------
- `z`  : 1-D numpy array of cross-sectional z-scores for a single date.
- `y`  : 1-D numpy array of *risk-normalised* returns (r_i / sigma_i) for that date.
- `update(z, y)` ingests one day's observations and advances the model state.
- `predict(z)` returns ``(alpha_z, variance_z)`` where:
      alpha_z   = IC(z) * z   (to be multiplied by sigma_i to get the Grinold alpha)
      variance_z = uncertainty of that prediction per stock
"""
from abc import ABC, abstractmethod
import numpy as np


class ICModel(ABC):
    """Base class for all IC function estimators."""

    @abstractmethod
    def update(self, z: np.ndarray, y: np.ndarray) -> None:
        """Incorporate one day's (z-score, risk-normalised return) pairs."""

    @abstractmethod
    def predict(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict alpha_z = IC(z)*z and its uncertainty for each z in *z*.

        Returns
        -------
        alpha_z : np.ndarray, shape (n,)
        variance_z : np.ndarray, shape (n,)
        """

    def reset(self) -> None:
        """Optionally reset model state (used between splits)."""

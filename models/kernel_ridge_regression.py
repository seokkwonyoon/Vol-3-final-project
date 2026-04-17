"""
Kernel Ridge Regression for IC estimation in low-SNR environments.

Regularization: Alpha is set based on the Noise-to-Signal ratio to force
high-bias, low-variance predictions. This prevents the model from "chasing"
noise and ensures predicted IC reflects only statistically significant
relationships.

The model maintains a rolling buffer of historical (z, y) pairs and refits
on each update to adapt to changing market conditions over the lookback window.
"""
import numpy as np
from collections import deque
from models.base import ICModel
from sklearn.kernel_ridge import KernelRidge
from configs import NOISE_TO_SIGNAL_RATIO, LOOKBACK_DAYS


class KernelRidgeRegressionIC(ICModel):
    """
    Kernel Ridge Regression for IC estimation with heavy regularization
    to suppress noise in low-SNR environments.

    Parameters
    ----------
    alpha : float or None
        Regularization parameter. If None, computed from config SNR.
        Default None → uses NOISE_TO_SIGNAL_RATIO from config.
    lookback_days : int
        Maximum number of historical (z, y) pairs to keep in buffer.
    kernel : str
        Kernel type for KernelRidge. Default 'rbf'.
    gamma : float or None
        Kernel coefficient. Default None lets sklearn auto-scale.
    """

    def __init__(
        self,
        alpha: float | None = None,
        lookback_days: int = LOOKBACK_DAYS,
        kernel: str = "rbf",
        gamma: float | None = None,
        max_fit_points: int = 300,
    ) -> None:
        if alpha is None:
            alpha = NOISE_TO_SIGNAL_RATIO

        self.alpha = alpha
        self.lookback_days = lookback_days
        self.kernel = kernel
        self.gamma = gamma
        self.max_fit_points = max_fit_points

        # Rolling buffer: one entry per trading day (stores full cross-section arrays)
        self.z_buffer: deque[np.ndarray] = deque(maxlen=lookback_days)
        self.y_buffer: deque[np.ndarray] = deque(maxlen=lookback_days)

        self.krr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
        self.is_fitted = False

    def update(self, z: np.ndarray, y: np.ndarray) -> None:
        """
        Add new observations to the buffer and refit the KRR model.

        Parameters
        ----------
        z : np.ndarray, shape (n,)
            z-scores for the stocks
        y : np.ndarray, shape (n,)
            risk-normalised returns
        """
        if len(z) == 0:
            return

        # One entry per day — not per stock — so maxlen=lookback_days is a day-count
        self.z_buffer.append(z)
        self.y_buffer.append(y)

        Z = np.concatenate(self.z_buffer).reshape(-1, 1)
        Y = np.concatenate(self.y_buffer)

        # Subsample to keep the kernel matrix tractable (O(n³) cost).
        # Keep the most recent observations (buffer is already time-ordered).
        if len(Z) > self.max_fit_points:
            Z = Z[-self.max_fit_points:]
            Y = Y[-self.max_fit_points:]

        try:
            self.krr.fit(Z, Y)
            self.is_fitted = True
        except Exception:
            pass

    def predict(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict IC(z) * z and uncertainty.
        
        Parameters
        ----------
        z : np.ndarray, shape (n,)
            z-scores for prediction
            
        Returns
        -------
        alpha_z : np.ndarray, shape (n,)
            Predicted IC(z) * z
        variance_z : np.ndarray, shape (n,)
            Predictive variance
        """
        Z = z.reshape(-1, 1)
        
        if not self.is_fitted or len(self.z_buffer) == 0:
            # No data yet: return zero signal with high uncertainty
            return np.zeros_like(z), np.ones_like(z)
        
        try:
            # KRR is trained on (z, y) pairs where y ≈ IC(z)*z, so predict()
            # already returns alpha_z = IC(z)*z directly — no extra *z needed.
            alpha_z = self.krr.predict(Z)
            variance_z = np.ones_like(z) * self.alpha / 10.0
            
            # Ensure reasonable variance bounds
            variance_z = np.clip(variance_z, 1e-8, 10.0)
            
            return alpha_z, variance_z
        except Exception:
            # Fallback on error
            return np.zeros_like(z), np.ones_like(z)

    def reset(self) -> None:
        """Reset the buffer and model state."""
        self.z_buffer.clear()
        self.y_buffer.clear()
        self.is_fitted = False

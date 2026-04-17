"""
Gaussian Process Regression for IC estimation in low-SNR environments.

Kernel: RBF + WhiteKernel (to capture noise explicitly)
The noise_level of WhiteKernel is fixed at 0.95 to represent the 95% noise
proportion, forcing the model to focus on the smoothest, most persistent
trends in the signal.

The model maintains a rolling buffer of historical (z, y) pairs and refits
on each update to adapt to changing market conditions over the lookback window.
"""
import numpy as np
from collections import deque
from models.base import ICModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from configs import NOISE_PROPORTION, LOOKBACK_DAYS


class GaussianProcessRegressionIC(ICModel):
    """
    Gaussian Process Regression for IC estimation.

    Parameters
    ----------
    noise_level : float
        Fixed noise level for WhiteKernel. Default 0.95 represents 95% noise.
    lookback_days : int
        Maximum number of historical (z, y) pairs to keep in buffer.
    alpha : float
        Small regularization for numerical stability.
    """

    def __init__(
        self,
        noise_level: float = NOISE_PROPORTION,
        lookback_days: int = LOOKBACK_DAYS,
        alpha: float = 1e-6,
        max_fit_points: int = 300,
    ) -> None:
        self.noise_level = noise_level
        self.lookback_days = lookback_days
        self.alpha = alpha
        self.max_fit_points = max_fit_points

        # Rolling buffer: one array per trading day, maxlen=lookback_days days
        self.z_buffer: deque = deque(maxlen=lookback_days)
        self.y_buffer: deque = deque(maxlen=lookback_days)
        
        # Initialize the GP model with RBF + WhiteKernel
        # The WhiteKernel's noise is fixed at noise_level
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
            noise_level=noise_level,
            noise_level_bounds="fixed"  # Fix the noise level
        )
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=None,  # Don't optimize hyperparameters
            normalize_y=True,
            n_restarts_optimizer=0,
        )
        
        self.is_fitted = False

    def update(self, z: np.ndarray, y: np.ndarray) -> None:
        """
        Add new observations to the buffer and refit the GP model.
        
        Parameters
        ----------
        z : np.ndarray, shape (n,)
            z-scores for the stocks
        y : np.ndarray, shape (n,)
            risk-normalised returns
        """
        if len(z) == 0:
            return
        
        # One entry per day so maxlen=lookback_days is a day count, not obs count
        self.z_buffer.append(z)
        self.y_buffer.append(y)

        Z = np.concatenate(list(self.z_buffer)).reshape(-1, 1)
        Y = np.concatenate(list(self.y_buffer))

        if len(Z) > self.max_fit_points:
            # Keep the most recent observations (buffer is already time-ordered).
            # Deterministic and maximally relevant vs random sampling.
            Z = Z[-self.max_fit_points:]
            Y = Y[-self.max_fit_points:]

        try:
            self.gp.fit(Z, Y)
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
            # GPR is trained on (z, y) pairs where y ≈ IC(z)*z, so predict()
            # already returns alpha_z = IC(z)*z directly — no extra *z needed.
            alpha_z, ic_std = self.gp.predict(Z, return_std=True)
            variance_z = np.maximum(ic_std ** 2, 1e-8)
            
            return alpha_z, variance_z
        except Exception:
            # Fallback on error
            return np.zeros_like(z), np.ones_like(z)

    def reset(self) -> None:
        """Reset the buffer and model state."""
        self.z_buffer.clear()
        self.y_buffer.clear()
        self.is_fitted = False

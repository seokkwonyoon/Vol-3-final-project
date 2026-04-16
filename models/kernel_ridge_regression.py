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
    ) -> None:
        # If alpha not specified, use the noise-to-signal ratio
        if alpha is None:
            alpha = NOISE_TO_SIGNAL_RATIO
        
        self.alpha = alpha
        self.lookback_days = lookback_days
        self.kernel = kernel
        self.gamma = gamma
        
        # Rolling buffer of (z, y) pairs
        self.z_buffer = deque(maxlen=lookback_days)
        self.y_buffer = deque(maxlen=lookback_days)
        
        # Initialize KernelRidge with RBF kernel
        self.krr = KernelRidge(
            alpha=alpha,
            kernel=kernel,
            gamma=gamma,
        )
        
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
        
        # Add new data to buffers
        self.z_buffer.extend(z)
        self.y_buffer.extend(y)
        
        # Refit the model with current buffer
        if len(self.z_buffer) > 0:
            Z = np.array(self.z_buffer).reshape(-1, 1)
            Y = np.array(self.y_buffer)
            
            try:
                self.krr.fit(Z, Y)
                self.is_fitted = True
            except Exception:
                # If fitting fails, keep the old model
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
            # Get prediction (IC(z))
            ic_z = self.krr.predict(Z)
            
            # alpha_z = IC(z) * z
            alpha_z = ic_z * z
            
            # For KRR, we estimate uncertainty via the dual coefficients
            # and the alpha regularization parameter
            # Uncertainty increases with regularization (higher noise hypothesis)
            # We use alpha as a proxy for prediction uncertainty
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

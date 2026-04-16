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
    ) -> None:
        self.noise_level = noise_level
        self.lookback_days = lookback_days
        self.alpha = alpha
        
        # Rolling buffer of (z, y) pairs
        self.z_buffer = deque(maxlen=lookback_days)
        self.y_buffer = deque(maxlen=lookback_days)
        
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
        
        # Add new data to buffers
        self.z_buffer.extend(z)
        self.y_buffer.extend(y)
        
        # Refit the model with current buffer
        if len(self.z_buffer) > 0:
            Z = np.array(self.z_buffer).reshape(-1, 1)
            Y = np.array(self.y_buffer)
            
            try:
                self.gp.fit(Z, Y)
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
            # Get mean prediction (IC(z)) and standard deviation
            ic_z, ic_std = self.gp.predict(Z, return_std=True)
            
            # alpha_z = IC(z) * z
            alpha_z = ic_z * z
            
            # Variance: (∂α/∂IC)² * Var[IC] + (∂α/∂z)² * Var[z]
            # Since IC and z are independent:
            # Var[IC*z] ≈ z² * Var[IC] + IC² * Var[z]
            # We approximate Var[z] as 0 (fixed observations)
            # So: variance_z ≈ z² * std[IC]²
            variance_z = (z ** 2) * (ic_std ** 2)
            
            # Ensure non-negative variance and handle numerical issues
            variance_z = np.maximum(variance_z, 1e-8)
            
            return alpha_z, variance_z
        except Exception:
            # Fallback on error
            return np.zeros_like(z), np.ones_like(z)

    def reset(self) -> None:
        """Reset the buffer and model state."""
        self.z_buffer.clear()
        self.y_buffer.clear()
        self.is_fitted = False

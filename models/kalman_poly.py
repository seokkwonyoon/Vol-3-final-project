"""
State-Space Polynomial Kalman Filter.

State:  β = [β0, β1, β2]  (polynomial IC coefficients)
IC(z)  = β0 + β1·z + β2·z²
alpha_z = IC(z) · z = β0·z + β1·z² + β2·z³

Observation model (one stock i):
  y_i = H_i · β + ε_i,   H_i = [z_i, z_i², z_i³],  ε_i ~ N(0, R)

State transition (random walk):
  β_{t+1} = β_t + w_t,   w_t ~ N(0, Q)

All stocks on a single day are treated as a batch observation.
"""
import numpy as np
from models.base import ICModel


class KalmanPolyIC(ICModel):
    """
    Parameters
    ----------
    poly_degree : int
        Degree of the IC polynomial (default 2 → β0 + β1*z + β2*z²).
    process_noise : float
        Diagonal entry of the process noise covariance Q.
    obs_noise : float
        Observation noise variance R (per stock).
    """

    def __init__(
        self,
        poly_degree: int = 2,
        process_noise: float = 1e-4,
        obs_noise: float = 1.0,
    ) -> None:
        self.d = poly_degree + 1          # state dimension
        self.Q = process_noise * np.eye(self.d)
        self.R = obs_noise

        # Initial state: prior centred on IC(z)=0.05 → β0=0.05 so alpha_z=0.05*z
        self.beta = np.zeros(self.d)
        self.beta[0] = 0.05               # β0 = 0.05 matches static baseline
        self.P = np.eye(self.d) * 1.0    # large initial uncertainty

    def _obs_matrix(self, z: np.ndarray) -> np.ndarray:
        """Build the observation matrix H of shape (n, d).
        Row i = [z_i, z_i², ..., z_i^{d}] (the basis for alpha_z = H·β).
        """
        n = len(z)
        H = np.empty((n, self.d))
        for k in range(self.d):
            H[:, k] = z ** (k + 1)       # powers z^1, z^2, ..., z^d
        return H

    def update(self, z: np.ndarray, y: np.ndarray) -> None:
        if len(z) == 0:
            return

        # Predict step (random walk state transition)
        P_pred = self.P + self.Q         # (d, d)

        # Information-filter batch update: O(d²n + d³) instead of O(n³).
        # When d=3 << n≈500 this is a ~20,000× speedup over the n×n solve.
        #
        # Information form:
        #   Lambda_pred = P_pred⁻¹                         (d×d)
        #   Lambda_new  = Lambda_pred + Hᵀ H / R            (d×d)
        #   eta_new     = Lambda_pred @ beta + Hᵀ y / R     (d,)
        #   P_new       = Lambda_new⁻¹,  beta_new = P_new @ eta_new
        H = self._obs_matrix(z)                    # (n, d)
        Lambda_pred = np.linalg.inv(P_pred)        # (d, d)
        Lambda_new = Lambda_pred + (H.T @ H) / self.R
        eta_new = Lambda_pred @ self.beta + (H.T @ y) / self.R
        self.P = np.linalg.inv(Lambda_new)
        self.beta = self.P @ eta_new

    def predict(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = self._obs_matrix(z)          # (n, d)
        alpha_z = H @ self.beta          # predicted alpha_z per stock
        # Predictive variance per stock: diag(H P H^T) + R
        HP = H @ self.P
        variance_z = np.einsum("ij,ij->i", HP, H) + self.R
        return alpha_z, variance_z

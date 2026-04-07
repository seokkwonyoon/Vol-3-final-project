"""
Recursive Basis Function Expansion with Recursive Least Squares (RBF-RLS).

Fixed Gaussian basis functions are spread across the z-score range.
The weight vector is updated online via Recursive Least Squares.

Model:
  φ_k(z) = exp( -(z - c_k)² / (2·σ_k²) )   (k = 1..K basis functions)
  f(z)   = Σ_k  w_k · φ_k(z)                (predicted alpha_z)

Training:  y_i ≈ f(z_i) + ε_i
"""
import numpy as np
from models.base import ICModel


class RbfRlsIC(ICModel):
    """
    Parameters
    ----------
    n_bases : int
        Number of Gaussian basis functions evenly spaced in [-3, 3].
    width : float
        Bandwidth of each Gaussian (σ_k).  Defaults to inter-centre spacing.
    forgetting : float
        RLS forgetting factor λ ∈ (0, 1].  1.0 = no forgetting.
    obs_noise : float
        Initial diagonal of the RLS covariance (P₀ = obs_noise · I).
    """

    def __init__(
        self,
        n_bases: int = 7,
        width: float | None = None,
        forgetting: float = 0.995,
        obs_noise: float = 1.0,
    ) -> None:
        self.centres = np.linspace(-3.0, 3.0, n_bases)
        self.width = width if width is not None else (6.0 / (n_bases - 1))
        self.lam = forgetting

        self.w = np.zeros(n_bases)
        self.w[n_bases // 2] = 0.05     # prior: IC ≈ 0.05 at z=0
        self.P = obs_noise * np.eye(n_bases)

    def _phi(self, z: np.ndarray) -> np.ndarray:
        """Feature matrix Φ of shape (n, K)."""
        diff = z[:, None] - self.centres[None, :]   # (n, K)
        return np.exp(-0.5 * (diff / self.width) ** 2)

    def update(self, z: np.ndarray, y: np.ndarray) -> None:
        if len(z) == 0:
            return
        Phi = self._phi(z)              # (n, K)

        # Vectorised RLS update (process all stocks simultaneously)
        # Equivalent to sequential update via matrix inversion lemma
        # P_new = (1/λ) * (P - P Φ^T (λI + Φ P Φ^T)^{-1} Φ P)
        lam = self.lam
        n = len(z)
        A = lam * np.eye(n) + Phi @ self.P @ Phi.T   # (n, n)
        K = self.P @ Phi.T @ np.linalg.inv(A)         # (K, n)
        self.w = self.w + K @ (y - Phi @ self.w)
        self.P = (self.P - K @ Phi @ self.P) / lam

    def predict(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Phi = self._phi(z)               # (n, K)
        alpha_z = Phi @ self.w
        # Predictive variance: diag(Φ P Φ^T)
        PhiP = Phi @ self.P
        variance_z = np.einsum("ij,ij->i", PhiP, Phi)
        return alpha_z, variance_z

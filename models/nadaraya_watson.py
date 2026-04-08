"""
Online Nadaraya-Watson Kernel Regression (incremental grid-based).

Maintains running weighted sums on a fixed z-score grid instead of a
raw observation buffer.  For a new z_query:

  f̂(z_query) = interp(z_grid,  Σ_i w_i·y_i / Σ_i w_i,  z_query)

where weights decay exponentially over time:

  w_i(t) = K_spatial(z_grid - z_i) · exp(-age_i / τ)

Each call to update() ages the existing sums by one day and accumulates
today's contribution in O(n_grid × n_obs) — roughly 200×700 ≈ 140 k ops.
Each call to predict() is O(n_query) (numpy interp).  This replaces the
original O(n_query × max_days × n_obs) ≈ 58 M ops per day formulation.
"""
import numpy as np
from models.base import ICModel


class NadarayaWatsonIC(ICModel):
    """
    Parameters
    ----------
    bandwidth_z : float
        Spatial bandwidth h_s for the Gaussian kernel.
    decay_days : float
        Temporal decay τ; a observation aged d days is down-weighted
        by exp(-d / τ).  Effectively replaces the hard max_days cutoff
        with a soft exponential one (data older than ~5τ has < 1 % weight).
    n_grid : int
        Number of evaluation points on the fixed z-grid [-3, 3].
    min_weight_sum : float
        Minimum Σw at a grid point required to make a prediction;
        below this the point falls back to IC = 0, variance = 1.
    """

    def __init__(
        self,
        bandwidth_z: float = 0.5,
        decay_days: float = 60.0,
        n_grid: int = 200,
        min_weight_sum: float = 1e-8,
    ) -> None:
        self.h_s    = bandwidth_z
        self.tau    = decay_days
        self.min_w  = min_weight_sum

        # Fixed evaluation grid
        self.z_grid = np.linspace(-3.0, 3.0, n_grid)

        # Running weighted sums on the grid
        self.sum_w   = np.zeros(n_grid)   # Σ k_s · k_t
        self.sum_wy  = np.zeros(n_grid)   # Σ k_s · k_t · y
        self.sum_wy2 = np.zeros(n_grid)   # Σ k_s · k_t · y²  (for variance)

        # Multiplicative per-day decay: k_t(age=1) = exp(-1 / τ)
        self._decay = np.exp(-1.0 / decay_days)

    def update(self, z: np.ndarray, y: np.ndarray) -> None:
        if len(z) == 0:
            return

        # Age all existing contributions by one trading day
        self.sum_w   *= self._decay
        self.sum_wy  *= self._decay
        self.sum_wy2 *= self._decay

        # Accumulate today's observations: spatial kernel (n_grid, n_obs)
        dz  = self.z_grid[:, None] - z[None, :]          # (n_grid, n_obs)
        k_s = np.exp(-0.5 * (dz / self.h_s) ** 2)        # (n_grid, n_obs)

        self.sum_w   += k_s.sum(axis=1)
        self.sum_wy  += (k_s * y[None, :]).sum(axis=1)
        self.sum_wy2 += (k_s * (y[None, :] ** 2)).sum(axis=1)

    def predict(self, z_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = self.sum_w > self.min_w

        # IC(z) on the grid: E[y | z]
        ic_grid = np.where(valid, self.sum_wy / self.sum_w, 0.0)

        # Variance on the grid: E[y²] - E[y]²
        ey2_grid  = np.where(valid, self.sum_wy2 / self.sum_w, 1.0)
        var_grid  = np.maximum(ey2_grid - ic_grid ** 2, 0.0)
        var_grid  = np.where(valid, var_grid, 1.0)

        # Interpolate to the actual query z-scores
        alpha_z    = np.interp(z_query, self.z_grid, ic_grid)
        variance_z = np.interp(z_query, self.z_grid, var_grid)

        # Fall back for queries where the grid has no data
        no_data = np.interp(z_query, self.z_grid, valid.astype(float)) < 0.5
        variance_z[no_data] = 1.0

        return alpha_z, variance_z

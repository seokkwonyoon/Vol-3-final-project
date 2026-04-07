"""
Dynamic Binned Kalman Filters.

Partitions z-scores into N equal-width bins. Each bin k has an independent
1-D Kalman filter tracking IC_k — the mean return per unit of z in that bin.

For stocks in bin k:
  y_i / z_i  ≈  IC_k  +  ε_i        (|z_i| > z_min to avoid division)
→ IC_k is the "IC" within that region of the z-score distribution.

Prediction:  alpha_z_i = IC_{k(z_i)} · z_i
"""
import numpy as np
from models.base import ICModel


class BinnedKalmanIC(ICModel):
    """
    Parameters
    ----------
    n_bins : int
        Number of z-score bins (evenly spaced in [z_min, z_max]).
    z_min, z_max : float
        Range of z-scores covered by the bins.
    process_noise : float
        Per-bin random-walk variance (Q_k).
    obs_noise : float
        Observation noise variance (R).
    z_floor : float
        Minimum |z| required to update a bin (avoids divide-by-near-zero).
    """

    def __init__(
        self,
        n_bins: int = 15,
        z_min: float = -3.0,
        z_max: float = 3.0,
        process_noise: float = 1e-4,
        obs_noise: float = 1.0,
        z_floor: float = 0.2,
    ) -> None:
        self.edges = np.linspace(z_min, z_max, n_bins + 1)
        self.centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.n_bins = n_bins
        self.Q = process_noise
        self.R = obs_noise
        self.z_floor = z_floor

        # Each bin: IC state and variance
        self.ic = np.full(n_bins, 0.05)   # prior: IC = 0.05
        self.p = np.ones(n_bins)           # prior variance

    def _bin_index(self, z: np.ndarray) -> np.ndarray:
        """Map z-scores to bin indices, clipping to valid range."""
        idx = np.searchsorted(self.edges, z, side="right") - 1
        return np.clip(idx, 0, self.n_bins - 1)

    def update(self, z: np.ndarray, y: np.ndarray) -> None:
        if len(z) == 0:
            return

        # 1. Predict step: all bins advance (add process noise)
        self.p = self.p + self.Q

        # 2. Filter stocks with |z| too small
        mask = np.abs(z) >= self.z_floor
        z_ok, y_ok = z[mask], y[mask]
        if len(z_ok) == 0:
            return

        # Observation for each stock: "IC estimate" = y_i / z_i
        ic_obs = y_ok / z_ok
        bins = self._bin_index(z_ok)

        # Update each bin's KF using the average observation in that bin
        for k in range(self.n_bins):
            sel = bins == k
            if not np.any(sel):
                continue
            y_bar = ic_obs[sel].mean()
            # Scalar KF update
            S = self.p[k] + self.R / sel.sum()
            gain = self.p[k] / S
            self.ic[k] = self.ic[k] + gain * (y_bar - self.ic[k])
            self.p[k] = (1 - gain) * self.p[k]

    def predict(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bins = self._bin_index(z)
        ic_vals = self.ic[bins]
        alpha_z = ic_vals * z
        variance_z = self.p[bins]
        return alpha_z, variance_z

    def get_ic_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (bin_centres, ic_values) for visualisation."""
        return self.centres.copy(), self.ic.copy()

"""
Online Nadaraya-Watson Kernel Regression.

Maintains a rolling buffer of (z, y) observations from the last
`max_days` trading days. For a new z_query:

  f̂(z) = Σ_i  w_i · y_i  /  Σ_i  w_i

where
  w_i = K_spatial(z_query - z_i) · K_temporal(t_query - t_i)
  K_spatial(Δz) = exp(-Δz² / (2·h_s²))
  K_temporal(Δt) = exp(-Δt / τ)            (Δt in trading days)
"""
import numpy as np
from collections import deque
from models.base import ICModel


class NadarayaWatsonIC(ICModel):
    """
    Parameters
    ----------
    bandwidth_z : float
        Spatial bandwidth h_s.
    decay_days : float
        Temporal decay τ (half-life in trading days).
    max_days : int
        Maximum number of days to keep in the buffer.
    min_weight_sum : float
        Minimum Σw required to make a prediction (otherwise fall back to 0).
    """

    def __init__(
        self,
        bandwidth_z: float = 0.5,
        decay_days: float = 60.0,
        max_days: int = 120,
        min_weight_sum: float = 1e-8,
    ) -> None:
        self.h_s = bandwidth_z
        self.tau = decay_days
        self.max_days = max_days
        self.min_w = min_weight_sum

        # Buffer: each element is (z_array, y_array, age_in_days)
        # age=0 → today, age=1 → yesterday, ...
        self._buf: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._day = 0   # current day counter

    def update(self, z: np.ndarray, y: np.ndarray) -> None:
        if len(z) == 0:
            return
        self._buf.appendleft((z.copy(), y.copy(), self._day))
        self._day += 1
        # Evict observations older than max_days
        while self._buf and (self._day - self._buf[-1][2]) > self.max_days:
            self._buf.pop()

    def predict(self, z_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._buf:
            return np.zeros_like(z_query), np.ones_like(z_query)

        # Concatenate buffer into arrays
        all_z = np.concatenate([b[0] for b in self._buf])
        all_y = np.concatenate([b[1] for b in self._buf])
        # Age in days: day of b relative to current day
        ages = np.concatenate([
            np.full(len(b[0]), self._day - b[2] - 1, dtype=float)
            for b in self._buf
        ])

        n_query = len(z_query)
        alpha_z = np.empty(n_query)
        variance_z = np.empty(n_query)

        # Batch computation over query points
        # spatial kernel: (n_query, n_buffer)
        dz = z_query[:, None] - all_z[None, :]      # (nq, nb)
        k_s = np.exp(-0.5 * (dz / self.h_s) ** 2)

        # temporal kernel: (n_buffer,) broadcast to (nq, nb)
        k_t = np.exp(-ages / self.tau)               # (nb,)

        w = k_s * k_t                                # (nq, nb)
        w_sum = w.sum(axis=1)                        # (nq,)

        valid = w_sum > self.min_w
        alpha_z[valid] = (w[valid] * all_y[None, :]).sum(axis=1)[valid] / w_sum[valid]
        alpha_z[~valid] = 0.0

        # Variance: weighted variance of y values
        if valid.any():
            alpha_z_v = alpha_z[valid]
            w_v = w[valid]
            w_sum_v = w_sum[valid]
            y_bar = alpha_z_v                        # (nv,)
            sq_dev = (all_y[None, :] - y_bar[:, None]) ** 2  # (nv, nb)
            variance_z[valid] = (w_v * sq_dev).sum(axis=1) / w_sum_v
        variance_z[~valid] = 1.0

        return alpha_z, variance_z

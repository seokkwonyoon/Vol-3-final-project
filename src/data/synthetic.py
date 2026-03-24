import numpy as np


def drifting_function(x, t, lam=1.05):
    return 0.5 * x - (lam ** (-t)) * (x - 2)


def generate_data(n_points=100, t=0, sigma=0.3, lam=1.05, seed=None):
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, n_points)
    y_true = drifting_function(x, t, lam=lam)
    noise = rng.normal(0, sigma, size=n_points)
    y_obs = y_true + noise
    return x, y_true, y_obs
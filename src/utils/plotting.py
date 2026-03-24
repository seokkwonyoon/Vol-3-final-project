import matplotlib.pyplot as plt


def plot_synthetic_data(x, y_true, y_obs, t):
    """
    Plot true function and noisy observations.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_true, label="True function")
    plt.scatter(x, y_obs, s=20, alpha=0.7, label="Noisy observations")
    plt.title(f"Synthetic data at time t = {t}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
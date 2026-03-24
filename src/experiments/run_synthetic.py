from src.data.synthetic import generate_data
from src.utils.plotting import plot_synthetic_data


def main():
    for t in [0, 5, 10, 20]:
        x, y_true, y_obs = generate_data(
            n_points=100,
            t=t,
            sigma=0.3,
            lam=1.05,
            seed=0
        )

        print(f"\nTime t = {t}")
        print("First 5 observed y values:", y_obs[:5])

        plot_synthetic_data(x, y_true, y_obs, t)


if __name__ == "__main__":
    main()
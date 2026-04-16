"""
Unit tests for all IC model implementations.

Tests use only numpy/synthetic data — no real market data required.
"""
import numpy as np
import pytest

from models import MODEL_REGISTRY, StaticIC, KalmanPolyIC, RbfRlsIC, BinnedKalmanIC, NadarayaWatsonIC
from models.base import ICModel
from configs import STATIC_IC


# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

def make_linear_data(n: int = 200, ic: float = 0.05, noise: float = 0.5):
    """Synthetic (z, y) pairs from the true linear model y = ic * z + noise."""
    z = RNG.standard_normal(n).clip(-3, 3)
    y = ic * z + RNG.normal(0, noise, n)
    return z, y


def make_nonlinear_data(n: int = 200, noise: float = 0.3):
    """True IC(z) = 0.05 + 0.02*z (monotone but non-constant)."""
    z = RNG.standard_normal(n).clip(-3, 3)
    true_ic = 0.05 + 0.02 * z
    y = true_ic * z + RNG.normal(0, noise, n)
    return z, y


# ── Registry and interface tests ──────────────────────────────────────────────

def test_registry_contains_all_models():
    assert set(MODEL_REGISTRY.keys()) == {"static", "kalman_poly", "rbf_rls",
                                           "binned_kalman", "nadaraya_watson"}


def test_all_models_are_icmodel_subclasses():
    for name, cls in MODEL_REGISTRY.items():
        assert issubclass(cls, ICModel), f"{name} must subclass ICModel"


def test_all_models_instantiate_with_defaults():
    for name, cls in MODEL_REGISTRY.items():
        model = cls()
        assert model is not None, f"{name} failed to instantiate"


# ── Output shape and type tests ───────────────────────────────────────────────

@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
def test_predict_output_shapes(model_name):
    model = MODEL_REGISTRY[model_name]()
    z_test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    alpha_z, variance = model.predict(z_test)
    assert alpha_z.shape == (5,), f"{model_name}: alpha_z shape mismatch"
    assert variance.shape == (5,), f"{model_name}: variance shape mismatch"


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
def test_predict_returns_finite_values(model_name):
    model = MODEL_REGISTRY[model_name]()
    z, y = make_linear_data(100)
    model.update(z, y)
    alpha_z, variance = model.predict(z)
    assert np.all(np.isfinite(alpha_z)), f"{model_name}: alpha_z has non-finite values"
    assert np.all(np.isfinite(variance)), f"{model_name}: variance has non-finite values"


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
def test_variance_is_non_negative(model_name):
    model = MODEL_REGISTRY[model_name]()
    z, y = make_linear_data(100)
    model.update(z, y)
    _, variance = model.predict(z)
    assert np.all(variance >= 0), f"{model_name}: negative variance"


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
def test_update_with_empty_arrays_does_not_crash(model_name):
    model = MODEL_REGISTRY[model_name]()
    model.update(np.array([]), np.array([]))   # should be a no-op
    z_test = np.array([-1.0, 0.0, 1.0])
    alpha_z, _ = model.predict(z_test)
    assert alpha_z.shape == (3,)


# ── Static IC tests ───────────────────────────────────────────────────────────

def test_static_ic_is_exactly_ic_times_z():
    model = StaticIC()
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    alpha_z, _ = model.predict(z)
    expected = STATIC_IC * z
    np.testing.assert_allclose(alpha_z, expected)


def test_static_ic_update_is_noop():
    model = StaticIC()
    z = np.array([1.0, 2.0])
    y = np.array([100.0, -100.0])
    model.update(z, y)   # should have no effect
    alpha_z, _ = model.predict(np.array([1.0]))
    assert abs(alpha_z[0] - STATIC_IC) < 1e-10


def test_static_ic_zero_variance():
    model = StaticIC()
    z = np.linspace(-3, 3, 50)
    _, variance = model.predict(z)
    np.testing.assert_array_equal(variance, np.zeros(50))


# ── Kalman Polynomial Filter tests ───────────────────────────────────────────

def test_kalman_poly_updates_state_toward_truth():
    """After many updates with linear data, IC estimate converges toward STATIC_IC."""
    model = KalmanPolyIC()
    rng = np.random.default_rng(0)
    for _ in range(500):
        z = rng.standard_normal(300).clip(-3, 3)
        y = STATIC_IC * z + rng.normal(0, 0.5, 300)
        model.update(z, y)

    # beta[0] is the coefficient of z^1 (the linear IC term); it should ≈ STATIC_IC
    assert abs(model.beta[0] - STATIC_IC) < 0.02, (
        f"KalmanPoly beta[0]={model.beta[0]:.4f} not close to {STATIC_IC}"
    )


def test_kalman_poly_predictive_variance_decreases_after_updates():
    """Uncertainty should decrease as data accumulates."""
    model = KalmanPolyIC()
    z_test = np.array([1.0])
    _, var_before = model.predict(z_test)

    rng = np.random.default_rng(0)
    for _ in range(100):
        z = rng.standard_normal(200).clip(-3, 3)
        y = STATIC_IC * z + rng.normal(0, 0.5, 200)
        model.update(z, y)

    _, var_after = model.predict(z_test)
    assert var_after[0] < var_before[0], "Variance should decrease after updates"


def test_kalman_poly_degree_affects_state_dimension():
    for degree in [1, 2, 3]:
        model = KalmanPolyIC(poly_degree=degree)
        assert model.beta.shape == (degree + 1,)
        assert model.P.shape == (degree + 1, degree + 1)


# ── RBF-RLS tests ─────────────────────────────────────────────────────────────

def test_rbf_rls_learns_positive_slope():
    """After updates, f(z) should be monotonically increasing (matches linear truth)."""
    model = RbfRlsIC()
    rng = np.random.default_rng(1)
    for _ in range(300):
        z = rng.standard_normal(300).clip(-3, 3)
        y = STATIC_IC * z + rng.normal(0, 0.4, 300)
        model.update(z, y)

    z_test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    alpha_z, _ = model.predict(z_test)
    # The predicted alpha should be monotonically increasing
    assert np.all(np.diff(alpha_z) >= 0), f"RBF-RLS not monotone: {alpha_z}"


def test_rbf_rls_basis_count():
    for n in [5, 7, 10]:
        model = RbfRlsIC(n_bases=n)
        assert len(model.centres) == n
        assert model.w.shape == (n,)


def test_rbf_rls_forgetting_factor_stored():
    model = RbfRlsIC(forgetting=0.99)
    assert model.lam == 0.99


# ── Binned Kalman tests ───────────────────────────────────────────────────────

def test_binned_kalman_bin_assignment():
    model = BinnedKalmanIC(n_bins=10, z_min=-3.0, z_max=3.0)
    z = np.array([-3.0, 0.0, 3.0])
    bins = model._bin_index(z)
    assert bins[0] == 0
    assert bins[1] == 4 or bins[1] == 5   # middle bins
    assert bins[2] == 9


def test_binned_kalman_ic_responds_to_data():
    """IC values in bins with data should update away from prior."""
    model = BinnedKalmanIC()
    prior = model.ic.copy()

    rng = np.random.default_rng(2)
    for _ in range(100):
        z = rng.uniform(-3, 3, 500)
        y = 0.1 * z + rng.normal(0, 0.3, 500)
        model.update(z, y)

    # At least some bins should have changed from prior
    assert not np.allclose(model.ic, prior), "IC values did not update"


def test_binned_kalman_get_ic_curve():
    model = BinnedKalmanIC(n_bins=15)
    centres, ic_vals = model.get_ic_curve()
    assert len(centres) == 15
    assert len(ic_vals) == 15


def test_binned_kalman_z_floor_prevents_update_on_near_zero():
    """Stocks with |z| < z_floor should not trigger bin updates."""
    model = BinnedKalmanIC(z_floor=0.5)
    prior = model.ic.copy()
    # All z-scores are near zero
    z = np.full(100, 0.1)
    y = np.full(100, 0.5)
    model.update(z, y)   # process noise will still apply
    # IC values should not have been pulled toward y (only process noise added)
    # The actual ic values may differ from prior only by process noise variance
    max_delta = np.abs(model.ic - prior).max()
    assert max_delta < 0.01, "z_floor not respected"


# ── Nadaraya-Watson tests ─────────────────────────────────────────────────────

def test_nadaraya_watson_empty_buffer_returns_zeros():
    model = NadarayaWatsonIC()
    z_test = np.array([-1.0, 0.0, 1.0])
    alpha_z, _ = model.predict(z_test)
    np.testing.assert_array_equal(alpha_z, np.zeros(3))


def test_nadaraya_watson_interpolates_observations():
    """With a single training point at z=1, y=0.1, prediction at z=1 ≈ 0.1."""
    model = NadarayaWatsonIC(bandwidth_z=0.5)
    model.update(np.array([1.0]), np.array([0.1]))
    alpha_z, _ = model.predict(np.array([1.0]))
    assert abs(alpha_z[0] - 0.1) < 1e-6


def test_nadaraya_watson_buffer_eviction():
    """Buffer should not grow beyond max_days entries."""
    model = NadarayaWatsonIC(max_days=5)
    for _ in range(20):
        model.update(np.array([1.0, -1.0]), np.array([0.05, -0.05]))
    assert len(model._buf) <= 5


def test_nadaraya_watson_temporal_decay():
    """Recent observations should outweigh older ones near their z value."""
    model = NadarayaWatsonIC(bandwidth_z=0.3, decay_days=5)

    # Old observation: z=1, y=0.10
    model.update(np.array([1.0]), np.array([0.10]))
    for _ in range(15):  # advance time
        model.update(np.array([0.0]), np.array([0.0]))

    # Recent observation: z=1, y=0.02
    model.update(np.array([1.0]), np.array([0.02]))

    alpha_z, _ = model.predict(np.array([1.0]))
    # Should be closer to 0.02 (recent) than 0.10 (old)
    assert abs(alpha_z[0] - 0.02) < abs(alpha_z[0] - 0.10), (
        f"Temporal decay not working: prediction={alpha_z[0]:.4f}"
    )


# ── Cross-model convergence test ──────────────────────────────────────────────

@pytest.mark.parametrize("model_name", ["kalman_poly", "rbf_rls", "binned_kalman", "nadaraya_watson"])
def test_dynamic_model_outperforms_zero_on_linear_data(model_name):
    """
    After training, predicted alpha at z=2 should be positive (consistent with
    true IC=0.05, y = 0.05*z data).
    """
    model = MODEL_REGISTRY[model_name]()
    rng = np.random.default_rng(7)
    for _ in range(200):
        z = rng.standard_normal(300).clip(-3, 3)
        y = 0.05 * z + rng.normal(0, 0.3, 300)
        model.update(z, y)

    alpha_z, _ = model.predict(np.array([2.0]))
    assert alpha_z[0] > 0, (
        f"{model_name}: alpha_z at z=2 should be positive, got {alpha_z[0]:.4f}"
    )

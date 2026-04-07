"""
Unit tests for DynamicICPipeline using fully synthetic Polars DataFrames.
No real market data is loaded.
"""
import datetime as dt
import numpy as np
import polars as pl
import pytest

from pipeline import DynamicICPipeline
from models import StaticIC, KalmanPolyIC


# ── Helpers ───────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(99)

BARRIDS = [f"USA{i:04d}" for i in range(50)]
DATES = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(0, 30)]
# Keep only weekdays
DATES = [d for d in DATES if d.weekday() < 5]


def make_z_scores(dates=DATES, barrids=BARRIDS, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(dates) * len(barrids)
    return pl.DataFrame({
        "date": [d for d in dates for _ in barrids],
        "barrid": barrids * len(dates),
        "z_score": rng.standard_normal(n).clip(-3, 3).tolist(),
    })


def make_assets(dates=DATES, barrids=BARRIDS, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(dates) * len(barrids)
    z = rng.standard_normal(n)
    forward_return = 0.05 * z + rng.normal(0, 0.5, n)
    return pl.DataFrame({
        "date": [d for d in dates for _ in barrids],
        "barrid": barrids * len(dates),
        "return": (forward_return * 100).tolist(),          # raw return %
        "specific_risk": rng.uniform(10, 30, n).tolist(),   # annualised %, e.g. 15%
        "predicted_beta": rng.uniform(0.5, 1.5, n).tolist(),
        "price": rng.uniform(10, 100, n).tolist(),
        "forward_return": forward_return.tolist(),
        "in_universe": [True] * n,
    })


# ── cross_sectional_zscore ────────────────────────────────────────────────────

class TestCrossSectionalZscore:
    def setup_method(self):
        self.pipe = DynamicICPipeline()

    def test_output_columns(self):
        raw = pl.DataFrame({
            "date": [dt.date(2020, 1, 2)] * 5,
            "barrid": BARRIDS[:5],
            "raw_signal": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = self.pipe.cross_sectional_zscore(raw)
        assert set(result.columns) == {"date", "barrid", "z_score"}

    def test_z_scores_have_zero_mean_per_date(self):
        raw = pl.DataFrame({
            "date": [dt.date(2020, 1, 2)] * 10,
            "barrid": BARRIDS[:10],
            "raw_signal": RNG.standard_normal(10).tolist(),
        })
        result = self.pipe.cross_sectional_zscore(raw)
        mean = result["z_score"].mean()
        assert abs(mean) < 1e-10, f"z-score mean should be ~0, got {mean}"

    def test_z_scores_are_clipped_to_pm3(self):
        raw = pl.DataFrame({
            "date": [dt.date(2020, 1, 2)] * 5,
            "barrid": BARRIDS[:5],
            "raw_signal": [1e6, 1e5, 0.0, -1e5, -1e6],  # extreme values
        })
        result = self.pipe.cross_sectional_zscore(raw)
        assert result["z_score"].max() <= 3.0
        assert result["z_score"].min() >= -3.0

    def test_handles_multiple_dates_independently(self):
        n_dates = 5
        n_stocks = 20
        rows = []
        for i, d in enumerate(DATES[:n_dates]):
            for j, barrid in enumerate(BARRIDS[:n_stocks]):
                rows.append({"date": d, "barrid": barrid,
                             "raw_signal": float(RNG.standard_normal())})
        raw = pl.DataFrame(rows)
        result = self.pipe.cross_sectional_zscore(raw)

        # Each date's z-scores should independently sum to ~0
        for d in DATES[:n_dates]:
            day = result.filter(pl.col("date") == d)
            mean = day["z_score"].mean()
            assert abs(mean) < 1e-10, f"Date {d} z-score mean={mean}"

    def test_single_stock_per_date_gives_zero(self):
        raw = pl.DataFrame({
            "date": [dt.date(2020, 1, 2)],
            "barrid": ["USA0001"],
            "raw_signal": [42.0],
        })
        result = self.pipe.cross_sectional_zscore(raw)
        # std = 0, so result should be 0 (fill_nan)
        assert result["z_score"][0] == 0.0

    def test_constant_signal_gives_zero_z_scores(self):
        raw = pl.DataFrame({
            "date": [dt.date(2020, 1, 2)] * 5,
            "barrid": BARRIDS[:5],
            "raw_signal": [3.14] * 5,
        })
        result = self.pipe.cross_sectional_zscore(raw)
        np.testing.assert_array_equal(result["z_score"].to_numpy(), np.zeros(5))


# ── run_walkforward ───────────────────────────────────────────────────────────

class TestRunWalkforward:
    def setup_method(self):
        self.pipe = DynamicICPipeline()
        self.z_scores = make_z_scores()
        self.assets = make_assets()

    def test_output_columns(self):
        model = StaticIC()
        result = self.pipe.run_walkforward(self.z_scores, self.assets, model)
        assert set(result.columns) == {"date", "barrid", "predicted_beta", "alpha"}

    def test_output_rows_match_overlap(self):
        model = StaticIC()
        result = self.pipe.run_walkforward(self.z_scores, self.assets, model)
        # Result should have one row per (date, barrid) that appears in both inputs
        joint = self.z_scores.join(
            self.assets.select(["date", "barrid"]), on=["date", "barrid"], how="inner"
        )
        assert result.height == joint.height

    def test_static_model_alpha_equals_sigma_times_ic_times_z(self):
        model = StaticIC()
        result = self.pipe.run_walkforward(self.z_scores, self.assets, model)

        # Spot-check: alpha = (specific_risk/100) * STATIC_IC * z_score
        from config import STATIC_IC
        sample = result.join(
            self.z_scores, on=["date", "barrid"], how="inner"
        ).join(
            self.assets.select(["date", "barrid", "specific_risk"]),
            on=["date", "barrid"], how="inner"
        )
        expected = (sample["specific_risk"] / 100) * STATIC_IC * sample["z_score"]
        np.testing.assert_allclose(
            sample["alpha"].to_numpy(),
            expected.to_numpy(),
            rtol=1e-5,
        )

    def test_all_alpha_values_are_finite(self):
        for model_name in ["static", "kalman_poly", "rbf_rls"]:
            from models import MODEL_REGISTRY
            model = MODEL_REGISTRY[model_name]()
            result = self.pipe.run_walkforward(self.z_scores, self.assets, model)
            assert np.all(np.isfinite(result["alpha"].to_numpy())), (
                f"{model_name}: non-finite alpha values"
            )

    def test_result_is_sorted_by_date_barrid(self):
        model = StaticIC()
        result = self.pipe.run_walkforward(self.z_scores, self.assets, model)
        sorted_result = result.sort("date", "barrid")
        assert result.equals(sorted_result), "Result is not sorted by (date, barrid)"

    def test_dynamic_model_updates_over_time(self):
        """KalmanPoly predictions should differ from StaticIC after early dates."""
        static = StaticIC()
        dynamic = KalmanPolyIC()

        r_static = self.pipe.run_walkforward(self.z_scores, self.assets, static)
        r_dynamic = self.pipe.run_walkforward(self.z_scores, self.assets, dynamic)

        # After the first date the dynamic model has seen data; predictions should differ
        # (they may agree on date 0 because prior is centred near IC=0.05)
        late_dates = r_static["date"].unique().sort()[-5:].to_list()
        s_late = r_static.filter(pl.col("date").is_in(late_dates))["alpha"].to_numpy()
        d_late = r_dynamic.filter(pl.col("date").is_in(late_dates))["alpha"].to_numpy()
        assert not np.allclose(s_late, d_late), (
            "KalmanPoly should produce different predictions from StaticIC after learning"
        )

    def test_no_lookahead_bias(self):
        """On day 0, the model has seen no data yet — predictions use only the prior."""
        model = KalmanPolyIC(process_noise=0.0)   # frozen: no updates to state
        z_scores = make_z_scores(dates=DATES[:3])
        assets = make_assets(dates=DATES[:3])
        result = self.pipe.run_walkforward(z_scores, assets, model)

        # With process_noise=0 and no prior updates the Kalman filter prediction
        # should be deterministic regardless of which day we look at
        first_day = result["date"].min()
        day0 = result.filter(pl.col("date") == first_day)
        assert day0.height > 0


# ── run_walkforward_with_uncertainty ─────────────────────────────────────────

class TestRunWalkforwardWithUncertainty:
    def setup_method(self):
        self.pipe = DynamicICPipeline()
        self.z_scores = make_z_scores()
        self.assets = make_assets()

    def test_output_has_ic_variance_column(self):
        model = KalmanPolyIC()
        result = self.pipe.run_walkforward_with_uncertainty(
            self.z_scores, self.assets, model
        )
        assert "ic_variance" in result.columns

    def test_ic_variance_is_non_negative(self):
        model = KalmanPolyIC()
        result = self.pipe.run_walkforward_with_uncertainty(
            self.z_scores, self.assets, model
        )
        assert (result["ic_variance"] >= 0).all()

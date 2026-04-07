"""
Sanity checks for config.py — values, path helpers, and consistency.
"""
import datetime as dt
import pytest

from config import (
    SPLITS, SIGNALS, MODELS, STATIC_IC, GAMMA, LOOKBACK_DAYS,
    MOMENTUM_WINDOW, MOMENTUM_SKIP, REVERSAL_WINDOW, MIN_PRICE,
    STYLE_FACTORS, INDUSTRY_FACTORS,
    z_scores_path, alphas_path, weights_dir, signal_name, split_dir,
    PROJECT_ROOT,
)


class TestSplits:
    def test_all_splits_present(self):
        assert set(SPLITS.keys()) == {"full", "train", "test", "anders"}

    def test_splits_have_start_and_end(self):
        for name, cfg in SPLITS.items():
            assert "start" in cfg, f"Split {name} missing start"
            assert "end" in cfg, f"Split {name} missing end"

    def test_split_dates_are_date_objects(self):
        for name, cfg in SPLITS.items():
            assert isinstance(cfg["start"], dt.date)
            assert isinstance(cfg["end"], dt.date)

    def test_split_start_before_end(self):
        for name, cfg in SPLITS.items():
            assert cfg["start"] < cfg["end"], f"Split {name}: start >= end"

    def test_train_test_are_contiguous(self):
        assert SPLITS["train"]["end"] == SPLITS["test"]["start"]

    def test_full_covers_train_and_test(self):
        assert SPLITS["full"]["start"] <= SPLITS["train"]["start"]
        assert SPLITS["full"]["end"] >= SPLITS["test"]["end"]


class TestSignalsAndModels:
    def test_signals_is_non_empty_list(self):
        assert isinstance(SIGNALS, list)
        assert len(SIGNALS) > 0

    def test_models_is_non_empty_list(self):
        assert isinstance(MODELS, list)
        assert len(MODELS) > 0

    def test_static_is_first_model(self):
        assert MODELS[0] == "static", "static baseline should be first in MODELS"

    def test_no_duplicate_signals(self):
        assert len(SIGNALS) == len(set(SIGNALS))

    def test_no_duplicate_models(self):
        assert len(MODELS) == len(set(MODELS))


class TestParameters:
    def test_static_ic_is_positive(self):
        assert STATIC_IC > 0

    def test_gamma_is_positive(self):
        assert GAMMA > 0

    def test_lookback_days_reasonable(self):
        assert 20 <= LOOKBACK_DAYS <= 504  # between 1 month and 2 years

    def test_momentum_window_greater_than_skip(self):
        assert MOMENTUM_WINDOW > MOMENTUM_SKIP

    def test_reversal_window_shorter_than_momentum(self):
        assert REVERSAL_WINDOW < MOMENTUM_WINDOW

    def test_min_price_positive(self):
        assert MIN_PRICE > 0


class TestFactorLists:
    def test_style_factors_non_empty(self):
        assert len(STYLE_FACTORS) > 0

    def test_industry_factors_non_empty(self):
        assert len(INDUSTRY_FACTORS) > 0

    def test_no_overlap_between_style_and_industry(self):
        overlap = set(STYLE_FACTORS) & set(INDUSTRY_FACTORS)
        assert len(overlap) == 0, f"Style/industry overlap: {overlap}"

    def test_no_duplicate_style_factors(self):
        assert len(STYLE_FACTORS) == len(set(STYLE_FACTORS))

    def test_no_duplicate_industry_factors(self):
        assert len(INDUSTRY_FACTORS) == len(set(INDUSTRY_FACTORS))


class TestPathHelpers:
    def test_split_dir_contains_project_root(self):
        path = split_dir("train")
        assert PROJECT_ROOT in path
        assert "train" in path

    def test_z_scores_path_contains_signal(self):
        path = z_scores_path("test", "style_momentum")
        assert "style_momentum" in path
        assert "z_scores" in path
        assert path.endswith(".parquet")

    def test_alphas_path_contains_signal_and_model(self):
        path = alphas_path("test", "style_momentum", "kalman_poly")
        assert "style_momentum" in path
        assert "kalman_poly" in path
        assert path.endswith(".parquet")

    def test_weights_dir_contains_gamma(self):
        path = weights_dir("test", "style_momentum", "static")
        assert str(GAMMA) in path

    def test_signal_name_combines_signal_and_model(self):
        name = signal_name("style_momentum", "kalman_poly")
        assert "style_momentum" in name
        assert "kalman_poly" in name

    @pytest.mark.parametrize("split", ["full", "train", "test", "anders"])
    def test_all_splits_produce_valid_paths(self, split):
        path = split_dir(split)
        assert isinstance(path, str)
        assert len(path) > 0

    @pytest.mark.parametrize("sig", ["style_momentum", "industry_momentum",
                                      "betting_against_beta"])
    @pytest.mark.parametrize("mod", ["static", "kalman_poly"])
    def test_path_helpers_consistent(self, sig, mod):
        a_path = alphas_path("test", sig, mod)
        w_dir = weights_dir("test", sig, mod)
        # alphas path should be inside the split dir
        assert split_dir("test") in a_path
        assert split_dir("test") in w_dir

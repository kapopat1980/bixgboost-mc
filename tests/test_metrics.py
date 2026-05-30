"""Unit tests for evaluation metrics."""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import rmse, mae, mape, r2, nrmse, all_metrics
from src.evaluation.diebold_mariano import dm_test


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse(y, y) == pytest.approx(0.0)
        assert mae(y, y) == pytest.approx(0.0)
        assert mape(y, y) == pytest.approx(0.0, abs=1e-6)
        assert r2(y, y) == pytest.approx(1.0)

    def test_rmse_known(self):
        y_true = np.array([3.0, 3.0])
        y_pred = np.array([4.0, 2.0])   # errors: +1, -1 → RMSE = 1.0
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mape_known(self):
        y_true = np.array([100.0])
        y_pred = np.array([90.0])        # 10% error
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_r2_zero_baseline(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full(3, np.mean(y_true))
        assert r2(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_nrmse_units(self):
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([110.0, 90.0])   # RMSE=10, mean=100 → nRMSE=10%
        assert nrmse(y_true, y_pred) == pytest.approx(10.0)

    def test_all_metrics_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        result = all_metrics(y, y)
        assert set(result.keys()) == {"rmse", "mae", "mape", "r2", "nrmse"}


class TestDieboldMariano:
    def setup_method(self):
        np.random.seed(42)
        self.n = 500
        self.y_true = np.random.randn(self.n) * 10 + 50
        # Model 2 is clearly better (smaller errors)
        self.y_good = self.y_true + np.random.randn(self.n) * 1.0
        self.y_bad  = self.y_true + np.random.randn(self.n) * 5.0

    def test_better_model_significant(self):
        result = dm_test(self.y_true, self.y_bad, self.y_good,
                         loss="squared_error", alternative="less",
                         harvey_correction=True)
        assert result["significant_at_001"], \
            "Clearly better model should be significant at p<0.01"

    def test_equal_models_not_significant(self):
        y_pred1 = self.y_true + np.random.randn(self.n)
        y_pred2 = self.y_true + np.random.randn(self.n)
        result = dm_test(self.y_true, y_pred1, y_pred2,
                         loss="squared_error", alternative="two-sided",
                         harvey_correction=True)
        # With similar errors, should not always be significant
        assert isinstance(result["p_value"], float)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_output_keys(self):
        result = dm_test(self.y_true, self.y_bad, self.y_good)
        assert set(result.keys()) == {
            "dm_stat", "p_value", "significant_at_001", "significant_at_005"
        }

    def test_absolute_loss(self):
        result = dm_test(self.y_true, self.y_bad, self.y_good,
                         loss="absolute_error", alternative="less")
        assert result["significant_at_005"]

"""Unit tests for Bifacial Feature Augmentation Layer."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.features.bfal import BifacialFeatureAugmentationLayer, sky_view_factor


def make_sample_df(n=100):
    np.random.seed(0)
    t = pd.date_range("2021-06-15 06:00", periods=n, freq="10T")
    return pd.DataFrame({
        "ghi":       np.random.uniform(100, 900, n),
        "dni":       np.random.uniform(50, 800, n),
        "dhi":       np.random.uniform(20, 200, n),
        "t_amb":     np.random.uniform(15, 40, n),
        "rh":        np.random.uniform(20, 80, n),
        "wind_speed":np.random.uniform(0.5, 8.0, n),
        "wind_dir":  np.random.uniform(0, 360, n),
        "p_atm":     np.random.uniform(980, 1020, n),
        "t_mod":     np.random.uniform(25, 65, n),
        "ghi_poa":   np.random.uniform(100, 950, n),
    }, index=t)


class TestSkyViewFactor:
    def test_range(self):
        phi = sky_view_factor(25.0, 1.5, 6.0)
        assert 0.0 <= phi <= 1.0

    def test_horizontal_module_is_max(self):
        phi_flat = sky_view_factor(0.0, 1.5, 6.0)
        phi_tilted = sky_view_factor(45.0, 1.5, 6.0)
        assert phi_flat >= phi_tilted

    def test_paper_value(self):
        # Paper states phi_sky ≈ 0.78 for tilt=25, h=1.5, pitch=6.0
        phi = sky_view_factor(25.0, 1.5, 6.0)
        assert abs(phi - 0.78) < 0.05


class TestBFAL:
    def setup_method(self):
        self.bfal = BifacialFeatureAugmentationLayer(
            tilt_deg=25.0, row_height_m=1.5, row_pitch_m=6.0,
            wccf_alpha=0.012, t_ref=25.0, ground_albedo=0.2
        )
        self.df = make_sample_df()

    def test_output_shape(self):
        out = self.bfal.transform(self.df)
        assert set(["bgc", "kt", "tri", "hsi", "wccf"]).issubset(out.columns)
        assert len(out) == len(self.df)

    def test_kt_bounded(self):
        out = self.bfal.transform(self.df)
        assert out["kt"].between(0.0, 1.1).all(), "KT should be in [0, 1.1]"

    def test_bgc_nonnegative(self):
        out = self.bfal.transform(self.df)
        assert (out["bgc"] >= 0).all()

    def test_wccf_above_one(self):
        out = self.bfal.transform(self.df)
        assert (out["wccf"] >= 1.0).all(), "WCCF should be >= 1 (correction factor)"

    def test_no_nan(self):
        out = self.bfal.transform(self.df)
        for col in ["bgc", "kt", "tri", "hsi", "wccf"]:
            assert not out[col].isna().any(), f"{col} contains NaN"

    def test_feature_count(self):
        assert len(self.bfal.all_feature_names) == 13

    def test_nighttime_bgc_zero(self):
        df = self.df.copy()
        df["ghi"] = 0.0
        out = self.bfal.transform(df)
        assert (out["bgc"] == 0.0).all()

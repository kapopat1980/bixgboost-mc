"""Unit tests for BiLSTMEncoder and BiXGBoost-MC pipeline shapes."""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.models.bilstm_encoder import BiLSTMEncoder, TemporalAttention
from src.models.xgboost_corrector import build_xrc_features
import pandas as pd


class TestTemporalAttention:
    def test_output_shapes(self):
        attn = TemporalAttention(hidden_dim=64)
        h = torch.randn(8, 24, 64)          # batch=8, seq=24, dim=64
        ctx, alpha = attn(h)
        assert ctx.shape == (8, 64)
        assert alpha.shape == (8, 24)

    def test_attention_sums_to_one(self):
        attn = TemporalAttention(hidden_dim=32)
        h = torch.randn(4, 10, 32)
        _, alpha = attn(h)
        sums = alpha.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)


class TestBiLSTMEncoder:
    def setup_method(self):
        self.model = BiLSTMEncoder(
            input_dim=13, hidden_dims=[128, 64], forecast_horizon=1, dropout=0.0
        )

    def test_forward_shapes(self):
        x = torch.randn(16, 24, 13)       # batch=16, seq=24, features=13
        y_hat, ctx, alpha = self.model(x)
        assert y_hat.shape == (16, 1)
        assert ctx.shape  == (16, 128)    # 2 * H2 = 2*64
        assert alpha.shape == (16, 24)

    def test_multi_horizon(self):
        model = BiLSTMEncoder(input_dim=13, hidden_dims=[64, 32], forecast_horizon=3)
        x = torch.randn(4, 24, 13)
        y_hat, _, _ = model(x)
        assert y_hat.shape == (4, 3)

    def test_no_nan_output(self):
        x = torch.randn(8, 24, 13)
        y_hat, ctx, alpha = self.model(x)
        assert not torch.isnan(y_hat).any()
        assert not torch.isnan(ctx).any()

    def test_parameter_count_reasonable(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        assert 500_000 < n_params < 5_000_000, \
            f"Unexpected parameter count: {n_params:,}"


class TestXRCFeatureBuilder:
    def test_output_shape(self):
        N = 200
        bfal = np.random.randn(N, 13)
        ctx  = np.random.randn(N, 128)
        pred = np.random.randn(N)
        ts   = pd.date_range("2021-01-01", periods=N, freq="10T")
        Z = build_xrc_features(bfal, ctx, pred, ts)
        assert Z.shape == (N, 148)

    def test_no_nan(self):
        N = 50
        Z = build_xrc_features(
            np.random.randn(N, 13),
            np.random.randn(N, 128),
            np.random.randn(N),
            pd.date_range("2021-06-01 08:00", periods=N, freq="10T"),
        )
        assert not np.isnan(Z).any()

    def test_sin_cos_bounded(self):
        N = 100
        ts = pd.date_range("2021-01-01", periods=N, freq="10T")
        Z = build_xrc_features(
            np.ones((N, 13)),
            np.ones((N, 128)),
            np.ones(N),
            ts,
        )
        # First four temporal features are sin/cos → must be in [-1, 1]
        temporal_cols = Z[:, 142:146]
        assert (temporal_cols >= -1.0).all() and (temporal_cols <= 1.0).all()

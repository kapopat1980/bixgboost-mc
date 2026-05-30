# Baseline Model Implementations

This document clarifies how each of the seven baseline models was implemented,
particularly how simpler models (Persistence, ARIMA) handled the 13-dimensional
input feature vector. This addresses the methodological note in Section 5.1 of
the paper.

---

## Feature Input Policy

All seven baselines receive the **same 13-dimensional input** (8 raw + 5 BFAL
features) as BiXGBoost-MC. This ensures comparisons are on an equal footing:
differences in performance reflect architectural capacity, not information
asymmetry.

For models that do not naturally accept a multi-variate feature vector, the
following adaptations were applied:

---

## Baseline 1 — Persistence Model

**Description:** ŷ_{t+h} = y_t (forecast equals the most recent observed value)

**Feature usage:** None. The Persistence model is a univariate, parameter-free
baseline that uses only the target variable. The 13-feature vector is not passed
to it. This is standard practice for Persistence baselines in solar forecasting
(e.g., Diagne et al., 2013).

**Justification:** The Persistence model serves as a lower bound — any
useful model should outperform it. Supplying additional features to a
persistence-style model would change it into a different model class.

---

## Baseline 2 — ARIMA(5,1,2)

**Description:** Autoregressive Integrated Moving Average model.

**Feature usage:** ARIMA is a univariate time-series model. It was trained
on the target variable time series only. The 13-feature vector was not used
as exogenous input.

**Justification:** ARIMAX (ARIMA with exogenous variables) was considered but
rejected to keep the baseline representative of the standard ARIMA literature
for solar forecasting. The intent is to benchmark classical univariate methods,
not to optimally tune a multivariate variant.

**Order selection:** The (5,1,2) order was determined by minimising AIC on the
training set via a grid search over p ∈ {1,...,8}, d ∈ {0,1}, q ∈ {0,...,5}.

---

## Baseline 3 — Random Forest

**Description:** Scikit-learn `RandomForestRegressor`

**Feature usage:** All 13 features used. A lookback window of L=24 steps was
flattened to create a 13×24=312-dimensional feature vector for each prediction
target (matching the same context window as the Bi-LSTM baselines).

**Hyperparameters:** n_estimators=500, max_features="sqrt", min_samples_leaf=5,
selected by 5-fold cross-validation on the training set.

---

## Baseline 4 — Standalone XGBoost

**Description:** XGBoost `XGBRegressor`

**Feature usage:** All 13 features used with the same 312-dim flattened lookback
vector as Random Forest.

**Hyperparameters:** Same grid as BiXGBoost-MC's XRC component (n_estimators=800,
max_depth=6, lr=0.05) to provide a fair comparison of the standalone XGBoost
versus the XGBoost-as-corrector role in BiXGBoost-MC.

---

## Baseline 5 — Vanilla LSTM

**Description:** Single-direction, single-layer LSTM

**Feature usage:** All 13 features, same sliding window sequence format as
BiXGBoost-MC (L=24 steps, 10-min resolution).

**Architecture:** hidden_size=256, 1 layer, dropout=0.30, decoder FC(256→128→K).
Trained with the same Adam optimiser, learning rate, and early stopping policy
as the BiLSTMEncoder to isolate the effect of bidirectionality and the XRC.

---

## Baseline 6 — CNN-LSTM

**Description:** 1-D Convolutional layer for local feature extraction, followed
by a unidirectional LSTM.

**Feature usage:** All 13 features in the same sliding window format.

**Architecture:** Conv1D(filters=64, kernel=3, padding=same) → MaxPool1D(2)
→ LSTM(hidden=128) → FC(128→K). Same training protocol as Vanilla LSTM.

---

## Baseline 7 — Transformer

**Description:** Encoder-only Transformer adapted from Wu et al. [11]
(Informer architecture, simplified for this task).

**Feature usage:** All 13 features, same sliding window format.

**Architecture:** 2 encoder layers, d_model=128, n_heads=8, FFN dim=256,
dropout=0.10. Positional encoding added to the input sequence.
Same training protocol as all deep learning baselines.

---

## Summary Table

| Baseline | Features used | Input format |
|----------|--------------|--------------|
| Persistence | Target only | Scalar (y_t) |
| ARIMA(5,1,2) | Target only | Time series |
| Random Forest | All 13 | Flattened window (312-dim) |
| XGBoost | All 13 | Flattened window (312-dim) |
| Vanilla LSTM | All 13 | Sequence (24 × 13) |
| CNN-LSTM | All 13 | Sequence (24 × 13) |
| Transformer | All 13 | Sequence (24 × 13) |
| **BiXGBoost-MC** | **All 13** | **Sequence (24 × 13)** |

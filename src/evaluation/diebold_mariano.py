"""
Diebold-Mariano (DM) test for comparing predictive accuracy.

Implements the Harvey-Leybourne-Newbold (1997) finite-sample correction
as used in the paper (Section 5.5, Table 7).

References
----------
Diebold, F.X. & Mariano, R.S. (2002). Comparing predictive accuracy.
  Journal of Business & Economic Statistics, 20(1), 134–144.
Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of
  prediction mean squared errors. International Journal of Forecasting, 13, 281–291.
"""

import numpy as np
from scipy import stats


def dm_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss: str = "squared_error",
    alternative: str = "less",
    harvey_correction: bool = True,
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test: is model 2 significantly more accurate than model 1?

    Parameters
    ----------
    y_true : ndarray (N,)
    y_pred1 : ndarray (N,)   baseline model predictions
    y_pred2 : ndarray (N,)   proposed model predictions
    loss : {'squared_error', 'absolute_error'}
    alternative : {'less', 'greater', 'two-sided'}
        'less'  → H1: model 2 is MORE accurate (loss2 < loss1)
    harvey_correction : bool
        Apply Harvey-Leybourne-Newbold finite-sample correction.
    h : int
        Forecast horizon (steps). Used in Harvey correction.

    Returns
    -------
    dict with keys: dm_stat, p_value, significant_at_001, significant_at_005
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    if loss == "squared_error":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute_error":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    T = len(d)
    d_bar = np.mean(d)

    # Newey-West long-run variance estimate
    gamma0 = np.var(d, ddof=1)
    lrv = gamma0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        lrv += 2 * (1 - k / h) * gamma_k
    lrv = max(lrv, 1e-10)

    dm_stat = d_bar / np.sqrt(lrv / T)

    if harvey_correction:
        # Harvey-Leybourne-Newbold correction factor
        correction = np.sqrt(
            (T + 1 - 2 * h + h * (h - 1) / T) / T
        )
        dm_stat = dm_stat * correction

    if alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(dm_stat)
    else:
        p_value = 2 * min(stats.norm.cdf(dm_stat), 1 - stats.norm.cdf(dm_stat))

    return {
        "dm_stat": round(float(dm_stat), 4),
        "p_value": float(p_value),
        "significant_at_001": bool(p_value < 0.01),
        "significant_at_005": bool(p_value < 0.05),
    }


def dm_table(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    proposed_key: str,
    **dm_kwargs,
) -> list[dict]:
    """
    Compute DM test of proposed model vs all baselines.

    Parameters
    ----------
    predictions : dict mapping model name → prediction array
    proposed_key : str   key of the proposed model in predictions

    Returns
    -------
    list of dicts (one row per baseline), suitable for pd.DataFrame
    """
    y_prop = predictions[proposed_key]
    rows = []
    for name, y_base in predictions.items():
        if name == proposed_key:
            continue
        result = dm_test(y_true, y_base, y_prop, **dm_kwargs)
        rows.append({"baseline": name, **result})
    return rows

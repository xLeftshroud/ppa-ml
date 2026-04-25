"""Evaluation metrics for PPA. Target is log1p(volume_in_litres); most metrics
are reported both in log-space and the raw-litre scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_arr(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def wmape(y_true, y_pred) -> float:
    """Weighted MAPE = sum(|err|) / sum(|y|). Primary headline metric.

    Also known as WAPE in some literature; WMAPE is the more common name."""
    y, yh = _to_arr(y_true), _to_arr(y_pred)
    denom = np.abs(y).sum()
    return float(np.abs(y - yh).sum() / denom) if denom > 0 else np.nan


def rmse(y_true, y_pred) -> float:
    y, yh = _to_arr(y_true), _to_arr(y_pred)
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mae(y_true, y_pred) -> float:
    y, yh = _to_arr(y_true), _to_arr(y_pred)
    return float(np.mean(np.abs(y - yh)))


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    y, yh = _to_arr(y_true), _to_arr(y_pred)
    return float(np.mean(np.abs((y - yh) / np.maximum(np.abs(y), eps))))


def smape(y_true, y_pred) -> float:
    y, yh = _to_arr(y_true), _to_arr(y_pred)
    denom = (np.abs(y) + np.abs(yh)) / 2
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y - yh) / denom))


def rmsle(y_true, y_pred) -> float:
    """Root Mean Squared Logarithmic Error. Inputs are on original (volume) scale.

    Formula: sqrt(mean((log1p(y) - log1p(max(0, yh)))^2)). Clipping negatives
    before log1p is the standard Kaggle / sklearn convention.
    """
    y, yh = _to_arr(y_true), _to_arr(y_pred)
    yh = np.clip(yh, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(yh)) ** 2)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    y, yh = _to_arr(y_true), _to_arr(y_pred)
    ss_res = float(np.sum((y - yh) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def mase(y_true, y_pred, y_train) -> float:
    """MASE with naive-1 scaling on training series."""
    y, yh, yt = _to_arr(y_true), _to_arr(y_pred), _to_arr(y_train)
    scale = np.mean(np.abs(np.diff(yt)))
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(y - yh)) / scale)


def metrics_table(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    train_time_sec: float | None = None,
) -> dict:
    """Compute full metric dict. Inputs are on log1p scale.

    Returns both log-space and original-scale metrics.
    """
    y_vol = np.expm1(np.asarray(y_true_log))
    yh_vol = np.expm1(np.asarray(y_pred_log))
    # guard against negative predictions after inverse transform
    yh_vol = np.clip(yh_vol, a_min=0.0, a_max=None)

    out = {
        "rmse": rmse(y_vol, yh_vol),
        "rmse_log": rmse(y_true_log, y_pred_log),
        "rmsle": rmsle(y_vol, yh_vol),
        "r2": r2_score(y_vol, yh_vol),
        "r2_log": r2_score(y_true_log, y_pred_log),
        "mape": mape(y_vol, yh_vol),
        "smape": smape(y_vol, yh_vol),
        "wmape": wmape(y_vol, yh_vol),
        "mae": mae(y_vol, yh_vol),
        "mae_log": mae(y_true_log, y_pred_log),
    }
    if train_time_sec is not None:
        out["train_time_sec"] = float(train_time_sec)
    return out


def stratified_wmape(
    df: pd.DataFrame,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    """Per-group WMAPE for error analysis (e.g. by customer, by pack_tier)."""
    y_vol = np.expm1(y_true_log)
    yh_vol = np.clip(np.expm1(y_pred_log), 0, None)
    tmp = pd.DataFrame({"g": df[group_col].values, "y": y_vol, "yh": yh_vol})
    out = (
        tmp.groupby("g")
        .apply(lambda s: pd.Series({"wmape": wmape(s.y, s.yh), "n": len(s)}))
        .reset_index()
        .rename(columns={"g": group_col})
    )
    return out

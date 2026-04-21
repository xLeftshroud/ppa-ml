"""Naive baselines for fair comparison floor.

Both baselines predict on log1p(volume) scale to match the ML models.
Missing history (e.g. SKU has no t-1 or t-52 value in train) falls back
to the per-SKU-customer mean of training log-volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PANEL_KEYS, TIME_COL


def _panel_series_map(df: pd.DataFrame, shift: int) -> pd.Series:
    """For each row, look up log_nielsen_total_volume at continuous_week - shift within same panel unit.

    Returns a Series aligned to df.index. NaN where not found.
    """
    sorted_df = df.sort_values(PANEL_KEYS + [TIME_COL]).copy()
    sorted_df["_target_cw"] = sorted_df[TIME_COL] - shift

    left = sorted_df[PANEL_KEYS + [TIME_COL, "_target_cw"]].reset_index()
    right = sorted_df[PANEL_KEYS + [TIME_COL, "log_nielsen_total_volume"]].rename(
        columns={TIME_COL: "_target_cw", "log_nielsen_total_volume": "_lag_val"}
    )
    merged = left.merge(right, on=PANEL_KEYS + ["_target_cw"], how="left")
    return merged.set_index("index")["_lag_val"].reindex(df.index)


def naive_predict(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """Predict y_t = y_{t-1} on log scale, within panel unit.

    Concatenates train + val for the shift lookup so that the first val
    week gets train's last week. Falls back to panel mean then global mean.
    """
    full = pd.concat([train_df, val_df], ignore_index=False)
    lagged = _panel_series_map(full, shift=1).loc[val_df.index]

    # fallbacks
    panel_mean = train_df.groupby(PANEL_KEYS)["log_nielsen_total_volume"].mean()
    global_mean = float(train_df["log_nielsen_total_volume"].mean())

    def _fill_row(row, lag_val):
        if pd.notna(lag_val):
            return lag_val
        key = tuple(row[k] for k in PANEL_KEYS)
        return panel_mean.get(key, global_mean)

    out = np.array(
        [
            _fill_row(r, lag_val)
            for r, lag_val in zip(val_df[PANEL_KEYS].to_dict("records"), lagged.values)
        ]
    )
    return out


def seasonal_naive_predict(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """Predict y_t = y_{t-52} on log scale (yearly seasonality)."""
    full = pd.concat([train_df, val_df], ignore_index=False)
    lagged = _panel_series_map(full, shift=52).loc[val_df.index]

    panel_mean = train_df.groupby(PANEL_KEYS)["log_nielsen_total_volume"].mean()
    global_mean = float(train_df["log_nielsen_total_volume"].mean())

    out = []
    for r, lag_val in zip(val_df[PANEL_KEYS].to_dict("records"), lagged.values):
        if pd.notna(lag_val):
            out.append(lag_val)
        else:
            key = tuple(r[k] for k in PANEL_KEYS)
            out.append(panel_mean.get(key, global_mean))
    return np.array(out)


BASELINES = {
    "naive": naive_predict,
    "seasonal_naive": seasonal_naive_predict,
}

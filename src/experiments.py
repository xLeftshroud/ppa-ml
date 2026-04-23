"""Multi-seed x multi-fold experiment runner.

For each of 5 seeds, for each of 4 CV folds, fits a model using the
provided builder and returns a long DataFrame of metrics. This is the
input to the statistical tests (Friedman / Nemenyi / Wilcoxon).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from .config import SEEDS
from .evaluate import metrics_table
from .split import expanding_window_cv


def run_across_seeds(
    build_model: Callable[[int], object],
    df_dev: pd.DataFrame,
    y_dev: np.ndarray,
    feature_cols: list[str],
    seeds: list[int] | None = None,
    passes_val: bool = False,
    model_name: str = "model",
    y_fit: np.ndarray | None = None,
    expects_raw: bool = False,
) -> pd.DataFrame:
    """Train `build_model(seed)` on each (seed, fold); return metrics DataFrame.

    `y_dev` is always log-scale (used by metrics_table). If the model was
    trained on raw volume (GLM objectives / XGB squaredlogerror), pass
    `y_fit=raw_volume_array` and `expects_raw=True`; the model's raw
    predictions are log1p'd before metrics so metrics_table sees log-space
    inputs and its raw-volume output columns stay comparable.
    """
    seeds = seeds or SEEDS
    folds = expanding_window_cv(df_dev)
    y_for_fit = y_fit if y_fit is not None else y_dev
    rows = []
    for seed in seeds:
        for fi, (tr_idx, va_idx) in enumerate(folds, start=1):
            X_tr = df_dev.iloc[tr_idx][feature_cols]
            y_tr = y_for_fit[tr_idx]
            X_va = df_dev.iloc[va_idx][feature_cols]
            y_va_fit = y_for_fit[va_idx]
            y_va_log = y_dev[va_idx]

            model = build_model(seed)
            t0 = time.perf_counter()
            if passes_val:
                model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va_fit)
            else:
                model.fit(X_tr, y_tr)
            train_time = time.perf_counter() - t0

            pred = model.predict(X_va)
            if expects_raw:
                pred = np.log1p(np.clip(pred, 0, None))
            m = metrics_table(y_va_log, pred, train_time_sec=train_time)
            m.update({"model": model_name, "seed": seed, "fold": fi})
            rows.append(m)
    return pd.DataFrame(rows)


def run_baseline_across_seeds(
    baseline_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    df_dev: pd.DataFrame,
    y_dev: np.ndarray,
    seeds: list[int] | None = None,
    model_name: str = "baseline",
) -> pd.DataFrame:
    """Baselines are deterministic, but we replicate them across seeds so they
    line up in the Friedman table with 20 observations per model."""
    seeds = seeds or SEEDS
    folds = expanding_window_cv(df_dev)
    rows = []
    base_cache: dict[int, dict] = {}

    for fi, (tr_idx, va_idx) in enumerate(folds, start=1):
        df_tr = df_dev.iloc[tr_idx]
        df_va = df_dev.iloc[va_idx]
        y_pred = baseline_fn(df_tr, df_va)
        t0 = time.perf_counter()
        _ = baseline_fn(df_tr, df_va)  # re-run to estimate time
        train_time = time.perf_counter() - t0
        base_cache[fi] = {
            "pred": y_pred, "y_true": y_dev[va_idx], "train_time_sec": train_time,
        }

    for seed in seeds:
        for fi, cached in base_cache.items():
            m = metrics_table(cached["y_true"], cached["pred"], cached["train_time_sec"])
            m.update({"model": model_name, "seed": seed, "fold": fi})
            rows.append(m)
    return pd.DataFrame(rows)


def summarize(metrics_df: pd.DataFrame, metric: str = "wmape") -> pd.DataFrame:
    """Mean + 95% CI across seeds*folds per model."""
    g = metrics_df.groupby("model")[metric]
    n = g.size()
    mean = g.mean()
    std = g.std(ddof=1)
    ci = 1.96 * std / np.sqrt(n)
    return (
        pd.DataFrame(
            {
                "mean": mean,
                "std": std,
                "ci95": ci,
                "n_obs": n,
            }
        )
        .sort_values("mean")
    )

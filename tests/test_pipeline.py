"""Smoke tests for the PPA pipeline.

Runs on a small subsample to verify imports + basic correctness.
Not a full accuracy test -- that's the job of the Optuna runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH
from src.features import build_features
from src.split import expanding_window_cv, final_holdout_split, describe_folds
from src.baselines import naive_predict, seasonal_naive_predict
from src.evaluate import wmape, metrics_table
from src.models.elastic_net import ElasticNetModel


def _small_df(n_sku: int = 30) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    sku_keep = df["product_sku_code"].drop_duplicates().head(n_sku)
    df = df[df["product_sku_code"].isin(sku_keep)].copy()
    return df


def test_feature_engineering():
    df = _small_df()
    df_fe = build_features(df)

    for col in ["log_nielsen_total_volume", "log_price_per_litre", "pack_size_total",
                "pack_tier"]:
        assert col in df_fe.columns, f"missing feature: {col}"
    from src.config import CANDIDATE_FEATURES
    assert "price_per_item" not in CANDIDATE_FEATURES
    assert "price_per_100ml" not in CANDIDATE_FEATURES
    assert (df_fe["log_nielsen_total_volume"] >= 0).all()
    print(f"[OK] feature_engineering: {df_fe.shape}")


def test_time_series_cv():
    df = _small_df()
    df_fe = build_features(df)

    dev_idx, test_idx = final_holdout_split(df_fe)
    assert len(dev_idx) > 0
    assert len(test_idx) >= 0
    df_dev = df_fe.iloc[dev_idx].reset_index(drop=True)

    folds = expanding_window_cv(df_dev)
    assert len(folds) >= 1, "expected at least one fold"
    # no overlap between train and val within a fold
    for tr, va in folds:
        assert len(set(tr) & set(va)) == 0
    summary = describe_folds(df_dev)
    print(f"[OK] time_series_cv: {len(folds)} folds")
    print(summary)


def test_baselines():
    df = _small_df()
    df_fe = build_features(df).dropna(subset=["log_nielsen_total_volume"]).reset_index(drop=True)

    folds = expanding_window_cv(df_fe)
    tr, va = folds[0]
    df_tr, df_va = df_fe.iloc[tr], df_fe.iloc[va]
    y_true = df_va["log_nielsen_total_volume"].values

    pred = naive_predict(df_tr, df_va)
    assert pred.shape == y_true.shape
    assert not np.isnan(pred).any()
    w_naive = wmape(np.expm1(y_true), np.expm1(pred))

    pred_s = seasonal_naive_predict(df_tr, df_va)
    assert pred_s.shape == y_true.shape
    w_sn = wmape(np.expm1(y_true), np.expm1(pred_s))
    print(f"[OK] baselines: naive WMAPE={w_naive:.3f} seasonal_naive WMAPE={w_sn:.3f}")


def test_elastic_net_fits():
    df = _small_df()
    df_fe = build_features(df).dropna(subset=["log_nielsen_total_volume"]).reset_index(drop=True)

    feats = ["log_price_per_litre", "promotion_indicator",
             "pack_size_total", "week_sin", "week_cos"]
    X = df_fe[feats].fillna(0.0)
    y = df_fe["log_nielsen_total_volume"].values

    m = ElasticNetModel(alpha=1e-2, l1_ratio=0.5, feature_cols=feats)
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == y.shape
    metrics = metrics_table(y, pred)
    elasticity = m.own_price_elasticity()
    print(f"[OK] elastic_net fit: WMAPE={metrics['wmape']:.3f}, "
          f"own-price elasticity={elasticity:.3f}")
    # sanity: elasticity should be negative-ish on real data (NOT an assert, just report)


def run_all():
    print(f"Running smoke tests against {DATA_PATH}\n")
    test_feature_engineering()
    print()
    test_time_series_cv()
    print()
    test_baselines()
    print()
    test_elastic_net_fits()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    run_all()

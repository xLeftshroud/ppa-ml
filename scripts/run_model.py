"""Unified runner for any of the 4 ML models (elastic_net/rf/xgb/lgb).

Usage (from project root):
    python -m scripts.run_model --model xgb --timeout 3600

Pipeline:
  1. Load + engineer features (leakage-free)
  2. Split: sealed 2025H2 test set; dev set -> 4 expanding-window folds
  3. Feature selection (domain core + VIF + BorutaShap + stability) on fold-1 train
  4. Optuna tuning on CV (wall-clock timeout)
  5. Refit best params across 5 seeds x 4 folds; save metrics_<model>.csv
  6. Evaluate on sealed test set; save elasticity_<model>.csv + model pickle

The Bayesian track has its own runner, `scripts/run_bayesian.py`.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_PATH,
    OUTPUTS,
    SEEDS,
    TUNING_WALLCLOCK_SEC,
    TUNING_MAX_TRIALS,
    CANDIDATE_FEATURES,
    CATEGORICAL_COLS,
)
from src.features import add_base_features, add_panel_features
from src.split import expanding_window_cv, final_holdout_split
from src.feature_selection import run_full_pipeline
from src.tuning import run_tuning
from src.experiments import run_across_seeds
from src.baselines import naive_predict, seasonal_naive_predict
from src.experiments import run_baseline_across_seeds
from src.evaluate import metrics_table
from src.elasticity import tree_local_elasticity, elastic_net_elasticity

from src.models.elastic_net import ElasticNetModel
from src.models.rf import RFModel
from src.models.xgb import XGBModel
from src.models.lgb import LGBModel


MODEL_CLASSES = {
    "elastic_net": ElasticNetModel,
    "rf": RFModel,
    "xgb": XGBModel,
    "lgb": LGBModel,
}
PASSES_VAL = {"elastic_net": False, "rf": False, "xgb": True, "lgb": True}


def load_and_engineer() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df_b = add_base_features(df)
    df_fe = add_panel_features(df_b, df_b)
    return df_fe


def pick_features(df_dev: pd.DataFrame, model_type: str) -> list[str]:
    """Run the 4-step feature selection. Returns final feature list."""
    # Elastic Net uses log_price_per_litre (log-log -> coef = elasticity);
    # tree models use raw price_per_litre.
    exclude = "price_per_litre" if model_type == "elastic_net" else "log_price_per_litre"
    candidates = [c for c in CANDIDATE_FEATURES
                  if c in df_dev.columns and c != exclude]
    cats_in = [c for c in CATEGORICAL_COLS if c in df_dev.columns]

    # take a smaller random sample for BorutaShap (performance)
    fs_df = df_dev.sample(min(10000, len(df_dev)), random_state=42).reset_index(drop=True)
    y_fs = fs_df["log_volume"].values
    result = run_full_pipeline(
        fs_df[candidates].dropna(),
        candidate_cols=candidates,
        y=y_fs,
        model_type=model_type,
        do_stability=False,  # turn to False if stability takes too long for quick run
    )
    final_numeric = result["final"]
    # always include categoricals for tree models
    if model_type in ("xgb", "lgb", "rf"):
        return list(dict.fromkeys(final_numeric + cats_in))
    return final_numeric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CLASSES.keys()))
    ap.add_argument("--timeout", type=int, default=TUNING_WALLCLOCK_SEC)
    ap.add_argument("--max-trials", type=int, default=TUNING_MAX_TRIALS,
                    help="Hard cap on Optuna trials; wall-clock usually triggers first.")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--skip-tune", action="store_true")
    ap.add_argument("--output-suffix", default="")
    ap.add_argument(
        "--metric",
        choices=["wmape", "rmse", "rmse_log", "rmsle", "mape", "smape", "r2", "r2_log"],
        default="rmse",
        help="Optuna tuning objective (minimize). r2/r2_log are negated internally.",
    )
    args = ap.parse_args()

    model_type = args.model
    model_cls = MODEL_CLASSES[model_type]
    passes_val = PASSES_VAL[model_type]

    print(f"[1/6] Loading data + engineering features...")
    df_fe = load_and_engineer()
    dev_idx, test_idx = final_holdout_split(df_fe)
    df_dev = df_fe.iloc[dev_idx].reset_index(drop=True)
    df_test = df_fe.iloc[test_idx].reset_index(drop=True)
    df_dev = df_dev.dropna(subset=["log_volume"]).reset_index(drop=True)
    y_dev = df_dev["log_volume"].values
    print(f"    dev rows={len(df_dev)}, sealed test rows={len(df_test)}")

    print(f"[2/6] Feature selection pipeline...")
    feature_cols = pick_features(df_dev, model_type)
    print(f"    {len(feature_cols)} features: {feature_cols}")

    folds = expanding_window_cv(df_dev)
    print(f"    CV folds: {len(folds)}")

    if not args.skip_tune:
        print(f"[3/6] Optuna TPE tuning ({args.timeout}s wall-clock)...")
        tune = run_tuning(
            model_type, df_dev, y_dev, folds, feature_cols,
            seed=args.seeds[0], timeout_sec=args.timeout,
            max_trials=args.max_trials,
            storage=f"sqlite:///{OUTPUTS / 'optuna.db'}",
            metric=args.metric,
        )
        print(f"    best {args.metric}={tune['best_value']:.4f} in {tune['n_trials']} trials")
        best_params = tune["best_params"]
    else:
        best_params = {}
        print("[3/6] Skipping tuning (--skip-tune)")

    print(f"[4/6] Refit across {len(args.seeds)} seeds x {len(folds)} folds...")

    def build(seed: int):
        return model_cls(**best_params, random_state=seed, feature_cols=feature_cols)

    metrics_df = run_across_seeds(
        build, df_dev, y_dev, feature_cols,
        seeds=args.seeds, passes_val=passes_val,
        model_name=model_type,
    )
    suffix = args.output_suffix
    metrics_path = OUTPUTS / f"metrics_{model_type}{suffix}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"    saved {metrics_path}")

    print(f"[5/6] Baselines on same folds...")
    for name, fn in [("naive", naive_predict), ("seasonal_naive", seasonal_naive_predict)]:
        out_path = OUTPUTS / f"metrics_{name}.csv"
        if not out_path.exists():
            bdf = run_baseline_across_seeds(
                fn, df_dev, y_dev, seeds=args.seeds, model_name=name,
            )
            bdf.to_csv(out_path, index=False)
            print(f"    saved {out_path}")
        else:
            print(f"    {out_path} already exists, skip")

    print(f"[6/6] Fit champion on all dev data + extract elasticity + test holdout...")
    champion = model_cls(**best_params, random_state=args.seeds[0], feature_cols=feature_cols)
    if passes_val:
        tr_idx, va_idx = folds[-1]
        champion.fit(
            df_dev.iloc[tr_idx][feature_cols], y_dev[tr_idx],
            X_val=df_dev.iloc[va_idx][feature_cols], y_val=y_dev[va_idx],
        )
    else:
        champion.fit(df_dev[feature_cols], y_dev)

    if model_type == "elastic_net":
        elast = elastic_net_elasticity(champion, df_dev)
    else:
        elast = tree_local_elasticity(champion, df_dev, feature_cols)
    elast_path = OUTPUTS / f"elasticity_{model_type}{suffix}.csv"
    elast.to_csv(elast_path, index=False)
    print(f"    saved {elast_path}")

    # final test evaluation
    if len(df_test):
        test_pred = champion.predict(df_test[feature_cols])
        test_metrics = metrics_table(df_test["log_volume"].values, test_pred)
        print(f"    sealed-test metrics: {test_metrics}")
        (OUTPUTS / f"test_metrics_{model_type}{suffix}.json").write_text(
            json.dumps(test_metrics, indent=2)
        )

    model_path = OUTPUTS / f"model_{model_type}{suffix}.joblib"
    meta_path = OUTPUTS / f"metadata_{model_type}{suffix}.json"

    joblib.dump(champion, model_path)

    versions = {
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if model_type == "xgb":
        import xgboost
        versions["xgboost"] = xgboost.__version__
    elif model_type == "lgb":
        import lightgbm
        versions["lightgbm"] = lightgbm.__version__

    metadata = {
        "model_type": model_type,
        "feature_cols": list(feature_cols),
        "best_params": {
            k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in best_params.items()
        },
        "train_end_week": int(df_dev["continuous_week"].max()),
        "n_train_rows": int(len(df_dev)),
        "seeds": list(args.seeds),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"    saved {model_path} + {meta_path}")
    print("done.")


if __name__ == "__main__":
    main()

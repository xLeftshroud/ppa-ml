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
from src.features import build_features
from src.split import expanding_window_cv, final_holdout_split
from src.feature_selection import run_full_pipeline
from src.tuning import run_tuning, XGB_MAX_ROUNDS, LGB_MAX_ROUNDS
from src.experiments import run_across_seeds
from src.baselines import naive_predict, seasonal_naive_predict
from src.experiments import run_baseline_across_seeds
from src.evaluate import metrics_table
from src.elasticity import tree_local_elasticity, elastic_net_elasticity

from src.models.elastic_net import ElasticNetModel
from src.models.rf import RFModel
from src.models.xgb import XGBModel
from src.models.lgb import LGBModel
from src.models.export import export_champion


MODEL_CLASSES = {
    "elastic_net": ElasticNetModel,
    "rf": RFModel,
    "xgb": XGBModel,
    "lgb": LGBModel,
}
PASSES_VAL = {"elastic_net": False, "rf": False, "xgb": True, "lgb": True}

# Which --objective values each model supports. Argparse accepts the union;
# we enforce per-model compatibility after parse.
_MODEL_OBJ_SUPPORTED = {
    "elastic_net": {"default", "poisson", "tweedie", "gamma"},
    "xgb":         {"default", "squaredlogerror", "poisson", "tweedie", "gamma"},
    "lgb":         {"default", "poisson", "tweedie", "gamma"},
    "rf":          {"default", "poisson", "gamma"},
}
_RAW_Y_OBJECTIVES = {"squaredlogerror", "poisson", "tweedie", "gamma"}


def load_and_engineer() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return build_features(df)


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
    y_fs = fs_df["log_nielsen_total_volume"].values
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
        choices=["rmse", "rmse_log", "rmsle", "r2", "r2_log",
                 "mape", "smape", "wmape", "mae", "mae_log"],
        default="rmse",
        help="Optuna tuning objective (minimize). r2/r2_log are negated internally.",
    )
    ap.add_argument(
        "--objective",
        default="default",
        choices=["default", "squaredlogerror", "poisson", "tweedie", "gamma"],
        help="Training loss. Per-model compatibility enforced after parse.",
    )
    args = ap.parse_args()

    model_type = args.model
    model_cls = MODEL_CLASSES[model_type]
    passes_val = PASSES_VAL[model_type]

    if args.objective not in _MODEL_OBJ_SUPPORTED[model_type]:
        raise SystemExit(
            f"{model_type} does not support --objective={args.objective}; "
            f"choose from {sorted(_MODEL_OBJ_SUPPORTED[model_type])}"
        )
    expects_raw = args.objective in _RAW_Y_OBJECTIVES
    objective_name = args.objective

    print(f"[1/6] Loading data + engineering features...")
    df_fe = load_and_engineer()
    dev_idx, test_idx = final_holdout_split(df_fe)
    df_dev = df_fe.iloc[dev_idx].reset_index(drop=True)
    df_test = df_fe.iloc[test_idx].reset_index(drop=True)
    df_dev = df_dev.dropna(subset=["nielsen_total_volume"]).reset_index(drop=True)
    y_dev = df_dev["log_nielsen_total_volume"].values            # always log-space (for metrics)
    y_raw = df_dev["nielsen_total_volume"].values.astype(float)  # raw volume (for GLM/squaredlogerror fit)
    y_fit = y_raw if expects_raw else y_dev
    if args.objective == "gamma":
        # GammaRegressor requires y > 0; nudge zeros up to avoid fit error.
        y_fit = np.maximum(y_fit, 1e-6)
    print(f"    dev rows={len(df_dev)}, sealed test rows={len(df_test)}")
    print(f"    objective={args.objective}, y_space_trained_on={'raw_volume' if expects_raw else 'log_volume'}")

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
            objective_name=objective_name,
            y_fit=y_fit, expects_raw=expects_raw,
        )
        print(f"    best {args.metric}={tune['best_value']:.4f} in {tune['n_trials']} trials")
        best_params = tune["best_params"]
    else:
        best_params = {}
        print("[3/6] Skipping tuning (--skip-tune)")

    print(f"[4/6] Refit across {len(args.seeds)} seeds x {len(folds)} folds...")

    def build(seed: int):
        return model_cls(
            **best_params, random_state=seed, objective=objective_name,
            feature_cols=feature_cols,
        )

    metrics_df = run_across_seeds(
        build, df_dev, y_dev, feature_cols,
        seeds=args.seeds, passes_val=passes_val,
        model_name=model_type,
        y_fit=y_fit, expects_raw=expects_raw,
    )
    obj_tag = "" if args.objective == "default" else f"_{args.objective}"
    suffix = f"{obj_tag}{args.output_suffix}"
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

    # Industry-standard refit: pick n_rounds* = median of per-fold best_iteration
    # (from early stopping with best_params), then refit on ALL dev data with
    # that fixed n_estimators and NO early stopping. Avoids losing the latest
    # weeks of data that a holdout-based refit would sacrifice.
    n_rounds_star: int | None = None
    if passes_val and not args.skip_tune:
        # Match the per-model ceiling used during tuning so early stopping
        # has the same headroom when we replay best_params to collect
        # best_iteration per fold.
        ceiling = XGB_MAX_ROUNDS if model_type == "xgb" else LGB_MAX_ROUNDS
        print(f"[5.5/6] Collecting best_iteration per fold using best params...")
        best_iters: list[int] = []
        for fi, (tr_idx, va_idx) in enumerate(folds, start=1):
            m_tmp = model_cls(
                **best_params,
                n_estimators=ceiling,
                random_state=args.seeds[0],
                objective=objective_name,
                feature_cols=feature_cols,
            )
            m_tmp.fit(
                df_dev.iloc[tr_idx][feature_cols], y_fit[tr_idx],
                X_val=df_dev.iloc[va_idx][feature_cols], y_val=y_fit[va_idx],
            )
            bi = getattr(m_tmp.est_, "best_iteration", None)
            if bi is None:
                bi = getattr(m_tmp.est_, "best_iteration_", None)
            bi = int(bi) if bi is not None else ceiling
            best_iters.append(bi)
            print(f"    fold {fi}: best_iteration={bi}")
        n_rounds_star = int(np.median(best_iters))
        print(f"    n_rounds* = median({best_iters}) = {n_rounds_star}")

    print(f"[6/6] Fit champion on all dev data + extract elasticity + test holdout...")
    champion_params = dict(best_params)
    if n_rounds_star is not None:
        champion_params["n_estimators"] = n_rounds_star
    champion = model_cls(
        **champion_params,
        random_state=args.seeds[0],
        objective=objective_name,
        feature_cols=feature_cols,
    )
    champion.fit(df_dev[feature_cols], y_fit)

    if model_type == "elastic_net":
        elast = elastic_net_elasticity(champion, df_dev)
    else:
        elast = tree_local_elasticity(
            champion, df_dev, feature_cols, predict_is_raw=expects_raw,
        )
    elast_path = OUTPUTS / f"elasticity_{model_type}{suffix}.csv"
    elast.to_csv(elast_path, index=False)
    print(f"    saved {elast_path}")

    # final test evaluation
    if len(df_test):
        test_pred = champion.predict(df_test[feature_cols])
        if expects_raw:
            # raw-volume pred -> log1p so metrics_table's internal expm1 recovers raw
            test_pred = np.log1p(np.clip(test_pred, 0, None))
        test_metrics = metrics_table(df_test["log_nielsen_total_volume"].values, test_pred)
        print(f"    sealed-test metrics: {test_metrics}")
        (OUTPUTS / f"test_metrics_{model_type}{suffix}.json").write_text(
            json.dumps(test_metrics, indent=2)
        )

    model_path = OUTPUTS / f"model_{model_type}{suffix}.joblib"
    meta_path = OUTPUTS / f"metadata_{model_type}{suffix}.json"

    # Save a self-contained sklearn object that returns raw nielsen_total_volume.
    # log-y model -> wrapped in TransformedTargetRegressor(log1p, expm1).
    # raw-y model (GLM/squaredlogerror) -> bare Pipeline (predict already raw).
    export_champion(champion, df_dev[feature_cols], y_raw, model_path)

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
        "objective": args.objective,
        "expects_raw_y": expects_raw,
        "y_space_trained_on": "raw_volume" if expects_raw else "log_volume",
        "feature_cols": list(feature_cols),
        "best_params": {
            k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in best_params.items()
        },
        "n_rounds_star": n_rounds_star,
        "train_end_week": int(df_dev["continuous_week"].max()),
        "n_train_rows": int(len(df_dev)),
        "seeds": list(args.seeds),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        # Contract for downstream app consumers:
        "target": "nielsen_total_volume",
        "target_transform": "log1p" if not expects_raw else "none",
        "predict_returns": "raw_volume",
        "schema_version": 2,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"    saved {model_path} + {meta_path}")
    print("done.")


if __name__ == "__main__":
    main()

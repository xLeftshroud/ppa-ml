"""Optuna TPE tuning with 1-hour wall-clock budget per model.

One unified entry: build_objective(model_type, df_dev, y_dev, folds, feature_cols, seed).
Returns a callable suitable for study.optimize(obj, timeout=3600).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .evaluate import wmape
from .models.elastic_net import ElasticNetModel
from .models.rf import RFModel
from .models.xgb import XGBModel
from .models.lgb import LGBModel


MODEL_TYPES = ("elastic_net", "rf", "xgb", "lgb")


def _mean_cv_wmape_with(model_builder, df_dev, y_dev, folds, feature_cols, passes_val=False):
    """Train model on each fold's train, predict on val, return mean WMAPE across folds.

    ``passes_val=True`` means the builder's fit() accepts X_val/y_val for early stopping.
    """
    vol_true = np.expm1(y_dev)
    scores = []
    for tr_idx, va_idx in folds:
        X_tr = df_dev.iloc[tr_idx][feature_cols]
        y_tr = y_dev[tr_idx]
        X_va = df_dev.iloc[va_idx][feature_cols]
        y_va = y_dev[va_idx]

        model = model_builder()
        if passes_val:
            model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        else:
            model.fit(X_tr, y_tr)
        pred_log = model.predict(X_va)
        pred_vol = np.clip(np.expm1(pred_log), 0, None)
        scores.append(wmape(vol_true[va_idx], pred_vol))
    return float(np.mean(scores))


def _suggest_elastic_net(trial, seed):
    return ElasticNetModel(
        alpha=trial.suggest_float("alpha", 1e-4, 1.0, log=True),
        l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
        random_state=seed,
    )


def _suggest_rf(trial, seed):
    return RFModel(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        random_state=seed,
    )


def _suggest_xgb(trial, seed):
    return XGBModel(
        n_estimators=2000,
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state=seed,
    )


def _suggest_lgb(trial, seed):
    return LGBModel(
        n_estimators=2000,
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", -1, 15),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 100),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state=seed,
    )


_SUGGESTERS = {
    "elastic_net": (_suggest_elastic_net, False),
    "rf":          (_suggest_rf,          False),
    "xgb":         (_suggest_xgb,         True),
    "lgb":         (_suggest_lgb,         True),
}


def build_objective(
    model_type: str,
    df_dev: pd.DataFrame,
    y_dev: np.ndarray,
    folds: list,
    feature_cols: list[str],
    seed: int = 42,
) -> Callable:
    """Return an Optuna objective closure for `model_type`.

    The objective reports mean WMAPE across all CV folds on the validation
    portion of each fold. Tuning then minimizes this value.
    """
    if model_type not in _SUGGESTERS:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Choose from {MODEL_TYPES}."
        )
    suggester, passes_val = _SUGGESTERS[model_type]

    def objective(trial) -> float:
        model_builder = lambda: suggester(trial, seed)
        return _mean_cv_wmape_with(
            model_builder, df_dev, y_dev, folds, feature_cols, passes_val=passes_val
        )

    return objective


def run_tuning(
    model_type: str,
    df_dev: pd.DataFrame,
    y_dev: np.ndarray,
    folds: list,
    feature_cols: list[str],
    seed: int = 42,
    timeout_sec: int = 3600,
    study_name: str | None = None,
    storage: str | None = None,
) -> dict:
    """Run a full Optuna study and return best params + study object."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name or f"{model_type}_seed{seed}",
        storage=storage,
        load_if_exists=True,
    )
    obj = build_objective(model_type, df_dev, y_dev, folds, feature_cols, seed=seed)
    study.optimize(obj, timeout=timeout_sec, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study": study,
    }

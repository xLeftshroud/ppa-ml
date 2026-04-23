"""Optuna TPE tuning with 1-hour wall-clock budget per model.

One unified entry: build_objective(model_type, df_dev, y_dev, folds, feature_cols, seed).
Returns a callable suitable for study.optimize(obj, timeout=3600).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .evaluate import metrics_table
from .models.elastic_net import ElasticNetModel
from .models.rf import RFModel
from .models.xgb import XGBModel
from .models.lgb import LGBModel


MODEL_TYPES = ("elastic_net", "rf", "xgb", "lgb")

# Per-model boosting-round ceiling during tuning. Early stopping inside each
# model's fit() picks the real best_iteration per fold -- these are just the
# upper bound. Calibrated to each model's learning_rate search range so even
# the lowest-lr trials can reach convergence before hitting the ceiling.
XGB_MAX_ROUNDS = 3000   # lr search [1e-3, 0.3]
LGB_MAX_ROUNDS = 3000   # lr search [1e-3, 0.3]
RF_MAX_ROUNDS = 1500    # lr search [1e-2, 0.3], HistGBR with internal early stopping

SUPPORTED_METRICS = (
    "rmse", "rmse_log", "rmsle", "r2", "r2_log",
    "mape", "smape", "wmape", "mae", "mae_log",
)
_MAXIMIZE = {"r2", "r2_log"}   # Optuna minimizes → negate these


def _mean_cv_score(
    trial, model_builder, df_dev, y_dev, folds, feature_cols,
    passes_val=False, metric="rmse",
    y_fit=None, expects_raw=False,
):
    """Train model on each fold's train, predict on val, return mean `metric`.

    `y_dev` is always log-scale (for metrics). When the model trains on raw
    volume (GLM objectives / XGB squaredlogerror), pass `y_fit=raw_y` and
    `expects_raw=True`. Raw predictions are log1p'd so metrics_table's
    internal expm1 recovers raw values for the output columns.

    r2 / r2_log are maximized → negated so Optuna can minimize.

    Reports the running mean score to Optuna after each fold so the study's
    pruner can kill obviously-bad trials before all folds run.
    """
    import optuna  # local import keeps tuning module import cheap

    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"unsupported metric: {metric}. Choose from {SUPPORTED_METRICS}"
        )
    y_for_fit = y_fit if y_fit is not None else y_dev
    scores = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        X_tr = df_dev.iloc[tr_idx][feature_cols]
        y_tr = y_for_fit[tr_idx]
        X_va = df_dev.iloc[va_idx][feature_cols]
        y_va_fit = y_for_fit[va_idx]
        y_va_log = y_dev[va_idx]

        model = model_builder()
        if passes_val:
            model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va_fit)
        else:
            model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        if expects_raw:
            pred = np.log1p(np.clip(pred, 0, None))
        m = metrics_table(y_va_log, pred)
        val = m[metric]
        scores.append(-val if metric in _MAXIMIZE else val)

        trial.report(float(np.mean(scores)), fi)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def _suggest_elastic_net(trial, seed, objective):
    return ElasticNetModel(
        alpha=trial.suggest_float("alpha", 1e-4, 1.0, log=True),
        l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
        random_state=seed,
        objective=objective,
    )


def _suggest_rf(trial, seed, objective):
    # max_iter fixed at ceiling; HistGBR's internal early stopping
    # (configured in RFModel.fit) picks the real round count per trial.
    return RFModel(
        max_iter=RF_MAX_ROUNDS,
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        random_state=seed,
        objective=objective,
    )


def _suggest_xgb(trial, seed, objective):
    return XGBModel(
        n_estimators=XGB_MAX_ROUNDS,
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state=seed,
        objective=objective,
    )


def _suggest_lgb(trial, seed, objective):
    return LGBModel(
        n_estimators=LGB_MAX_ROUNDS,
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", -1, 15),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 100),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state=seed,
        objective=objective,
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
    metric: str = "rmse",
    objective_name: str = "default",
    y_fit: np.ndarray | None = None,
    expects_raw: bool = False,
) -> Callable:
    """Return an Optuna objective closure for `model_type`.

    The objective reports mean `metric` across all CV folds on the validation
    portion of each fold. Tuning then minimizes this value (r2 is negated).

    `objective_name` is forwarded to each wrapper so trials use the chosen
    loss (default / poisson / tweedie / gamma / squaredlogerror).
    """
    if model_type not in _SUGGESTERS:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Choose from {MODEL_TYPES}."
        )
    suggester, passes_val = _SUGGESTERS[model_type]

    def objective(trial) -> float:
        model_builder = lambda: suggester(trial, seed, objective_name)
        return _mean_cv_score(
            trial, model_builder, df_dev, y_dev, folds, feature_cols,
            passes_val=passes_val, metric=metric,
            y_fit=y_fit, expects_raw=expects_raw,
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
    max_trials: int | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    metric: str = "rmse",
    objective_name: str = "default",
    y_fit: np.ndarray | None = None,
    expects_raw: bool = False,
) -> dict:
    """Run a full Optuna study and return best params + study object."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    # Fold-level MedianPruner: after each CV fold, if the running mean is
    # worse than the median of completed trials at the same fold index, the
    # trial is pruned. n_startup_trials=5 collects baseline stats before
    # pruning kicks in; n_warmup_steps=1 gives every trial at least one fold.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name or f"{model_type}_{objective_name}_{metric}_seed{seed}",
        storage=storage,
        load_if_exists=True,
    )
    obj = build_objective(
        model_type, df_dev, y_dev, folds, feature_cols,
        seed=seed, metric=metric,
        objective_name=objective_name, y_fit=y_fit, expects_raw=expects_raw,
    )
    study.optimize(
        obj,
        timeout=timeout_sec,
        n_trials=max_trials,
        show_progress_bar=False,
    )
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study": study,
    }

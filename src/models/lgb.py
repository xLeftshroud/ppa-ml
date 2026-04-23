"""LightGBM regressor wrapped in a sklearn Pipeline with OHE + TargetEncoder.

`native categorical` is NOT used here -- cats are OHE/TE encoded upstream
so the Pipeline can be pickled and reloaded with only sklearn/lgb installed.
Monotone constraint on price_per_litre is set after prep.fit discovers the
post-encoding column order (LightGBM 4.x only accepts list form, not dict).

Objective is switchable via `objective`:
- `default` → regression (L2) on log-y
- `poisson` → poisson on raw-y (log-link)
- `tweedie` → tweedie on raw-y (log-link, power=1.5)
- `gamma`   → gamma on raw-y (log-link, y>0)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .preprocess import build_encoder
from ..config import CATEGORICAL_COLS

MONOTONIC_PRICE_FEAT = "price_per_litre"
_TWEEDIE_POWER = 1.5

_OBJ_MAP = {
    "default": "regression",
    "poisson": "poisson",
    "tweedie": "tweedie",
    "gamma":   "gamma",
}
_RAW_Y_OBJECTIVES = {"poisson", "tweedie", "gamma"}


@dataclass
class LGBModel:
    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    n_estimators: int = 1000
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 50
    objective: str = "default"
    feature_cols: list[str] | None = None

    def __post_init__(self):
        if self.objective not in _OBJ_MAP:
            raise ValueError(
                f"objective must be one of {tuple(_OBJ_MAP)}, got {self.objective!r}"
            )
        self.pipeline_: Pipeline | None = None
        self.est_ = None
        self._feature_order_: list[str] | None = None

    @property
    def expects_raw_y(self) -> bool:
        return self.objective in _RAW_Y_OBJECTIVES

    def _split_cols(self, cols: list[str]) -> tuple[list[str], list[str]]:
        cats = [c for c in CATEGORICAL_COLS if c in cols]
        nums = [c for c in cols if c not in cats]
        return nums, cats

    def _build(self, X: pd.DataFrame) -> Pipeline:
        import lightgbm as lgb

        cols = self.feature_cols or list(X.columns)
        nums, cats = self._split_cols(cols)
        prep = build_encoder(
            X[cols], cat_cols=cats, num_cols=nums,
            high_card_threshold=20, scale_numeric=False,
        )
        kw = dict(
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_data_in_leaf=self.min_data_in_leaf,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective=_OBJ_MAP[self.objective],
            # monotone_constraints is set after prep is fit so we know the
            # post-encoding column order (LightGBM expects list form).
            monotone_constraints_method="intermediate",
            verbose=-1,
        )
        if self.objective == "tweedie":
            kw["tweedie_variance_power"] = _TWEEDIE_POWER
        model = lgb.LGBMRegressor(**kw)
        return Pipeline([("prep", prep), ("model", model)])

    @staticmethod
    def _monotone_list(feature_names: list[str]) -> list[int]:
        return [-1 if f == MONOTONIC_PRICE_FEAT else 0 for f in feature_names]

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LGBModel":
        import lightgbm as lgb

        cols = self.feature_cols or list(X.columns)
        X = X[cols]
        self.pipeline_ = self._build(X)
        prep = self.pipeline_.named_steps["prep"]
        model = self.pipeline_.named_steps["model"]

        prep.fit(X, y)
        feat_names = list(prep.get_feature_names_out())
        model.set_params(monotone_constraints=self._monotone_list(feat_names))
        Xp = prep.transform(X)

        if X_val is not None and y_val is not None:
            Xvp = prep.transform(X_val[cols])
            model.fit(
                Xp, y,
                eval_set=[(Xvp, y_val)],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
        else:
            model.fit(Xp, y)

        self.est_ = model
        self._feature_order_ = feat_names
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_cols or list(X.columns)
        return self.pipeline_.predict(X[cols])

    @property
    def feature_importance_(self) -> pd.Series:
        return pd.Series(
            self.est_.feature_importances_, index=self._feature_order_
        ).sort_values(ascending=False)

"""XGBoost regressor with monotonic constraint on price_per_litre.

Wrapped in a sklearn Pipeline whose preprocessor uses OneHot + TargetEncoder
(no native categorical). The monotone constraint is anchored to the column
name "price_per_litre" via XGB's dict form, which survives preprocessing
because `verbose_feature_names_out=False` preserves numeric column names.

Objective is switchable via `objective`:
- `default`         → reg:squarederror on log-y
- `squaredlogerror` → reg:squaredlogerror on raw-y (XGB log1p's internally)
- `poisson`         → count:poisson on raw-y (log-link)
- `tweedie`         → reg:tweedie on raw-y (log-link, power=1.5)
- `gamma`           → reg:gamma on raw-y (log-link, y>0)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from .preprocess import build_encoder
from ..config import CATEGORICAL_COLS

MONOTONIC_PRICE_FEAT = "price_per_litre"
_TWEEDIE_POWER = 1.5

_OBJ_MAP = {
    "default":         "reg:squarederror",
    "squaredlogerror": "reg:squaredlogerror",
    "poisson":         "count:poisson",
    "tweedie":         "reg:tweedie",
    "gamma":           "reg:gamma",
}
_RAW_Y_OBJECTIVES = {"squaredlogerror", "poisson", "tweedie", "gamma"}


@dataclass
class XGBModel:
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: float = 5.0
    gamma: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
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
        self.est_: xgb.XGBRegressor | None = None
        self._feature_order_: list[str] | None = None

    @property
    def expects_raw_y(self) -> bool:
        return self.objective in _RAW_Y_OBJECTIVES

    def _split_cols(self, cols: list[str]) -> tuple[list[str], list[str]]:
        cats = [c for c in CATEGORICAL_COLS if c in cols]
        nums = [c for c in cols if c not in cats]
        return nums, cats

    def _build(self, X: pd.DataFrame) -> Pipeline:
        cols = self.feature_cols or list(X.columns)
        nums, cats = self._split_cols(cols)
        prep = build_encoder(
            X[cols], cat_cols=cats, num_cols=nums,
            high_card_threshold=20, scale_numeric=False,
        )
        kw = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            objective=_OBJ_MAP[self.objective],
            monotone_constraints={MONOTONIC_PRICE_FEAT: -1},
        )
        if self.objective == "tweedie":
            kw["tweedie_variance_power"] = _TWEEDIE_POWER
        model = xgb.XGBRegressor(**kw)
        return Pipeline([("prep", prep), ("model", model)])

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBModel":
        cols = self.feature_cols or list(X.columns)
        X = X[cols]
        self.pipeline_ = self._build(X)
        prep = self.pipeline_.named_steps["prep"]
        model = self.pipeline_.named_steps["model"]

        if X_val is not None and y_val is not None:
            # Fit preprocessor on train, transform both sets, then fit XGB
            # separately so eval_set can be passed with early stopping.
            prep.fit(X, y)
            Xp = prep.transform(X)
            Xvp = prep.transform(X_val[cols])
            model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            model.fit(Xp, y, eval_set=[(Xvp, y_val)], verbose=False)
        else:
            self.pipeline_.fit(X, y)

        self.est_ = model
        self._feature_order_ = list(prep.get_feature_names_out())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_cols or list(X.columns)
        return self.pipeline_.predict(X[cols])

    @property
    def feature_importance_(self) -> pd.Series:
        return pd.Series(
            self.est_.feature_importances_, index=self._feature_order_
        ).sort_values(ascending=False)

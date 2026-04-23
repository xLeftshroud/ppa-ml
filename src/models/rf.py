"""HistGradientBoosting regressor wrapped in a sklearn Pipeline.

Class name `RFModel` is preserved so the training/tuning code keeps
working, but the underlying estimator is HistGradientBoostingRegressor
(sklearn >= 1.5 for dict-form monotonic_cst). Cats go through OHE/TE,
not native categorical splits.

Loss is switchable via `objective`:
- `default` → squared_error on log-y
- `poisson` → poisson on raw-y (log-link)
- `gamma`   → gamma on raw-y (log-link, y>0)

HGB does not support tweedie or squaredlogerror.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from .preprocess import build_encoder
from ..config import CATEGORICAL_COLS

MONOTONIC_PRICE_FEAT = "price_per_litre"

_LOSS_MAP = {
    "default": "squared_error",
    "poisson": "poisson",
    "gamma":   "gamma",
}
_RAW_Y_OBJECTIVES = {"poisson", "gamma"}


@dataclass
class RFModel:
    max_iter: int = 1500
    max_depth: int | None = None
    min_samples_leaf: int = 20
    learning_rate: float = 0.1
    l2_regularization: float = 0.0
    random_state: int = 42
    objective: str = "default"
    feature_cols: list[str] | None = None

    def __post_init__(self):
        if self.objective not in _LOSS_MAP:
            raise ValueError(
                f"objective must be one of {tuple(_LOSS_MAP)}, got {self.objective!r}"
            )
        self.pipeline_: Pipeline | None = None
        self.est_: HistGradientBoostingRegressor | None = None
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
        model = HistGradientBoostingRegressor(
            loss=_LOSS_MAP[self.objective],
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            monotonic_cst={MONOTONIC_PRICE_FEAT: -1},
            early_stopping=True,
            n_iter_no_change=50,
            validation_fraction=0.1,
            random_state=self.random_state,
        )
        return Pipeline([("prep", prep), ("model", model)])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RFModel":
        cols = self.feature_cols or list(X.columns)
        X = X[cols]
        self.pipeline_ = self._build(X)
        self.pipeline_.fit(X, y)
        self.est_ = self.pipeline_.named_steps["model"]
        self._feature_order_ = list(
            self.pipeline_.named_steps["prep"].get_feature_names_out()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_cols or list(X.columns)
        return self.pipeline_.predict(X[cols])

    @property
    def feature_importance_(self) -> pd.Series:
        """HistGBR does not expose built-in importances; zeros keep
        downstream fallbacks from crashing. Use permutation importance
        for a real ranking."""
        n = len(self._feature_order_ or [])
        return pd.Series(np.zeros(n), index=self._feature_order_ or [])

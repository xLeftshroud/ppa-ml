"""RandomForestRegressor wrapped in a sklearn Pipeline with OHE + TargetEncoder.

Cats use the same OHE/TE dispatch as XGB/LGB/HGB so the saved model is a pure
sklearn object. Monotone constraint on `price_per_litre` is set after prep.fit
discovers the post-encoding column order (sklearn RF accepts list form only,
like LightGBM).

Loss is switchable via `objective`:
- `squared_error` -> MSE on log-y
- `poisson`       -> Poisson criterion on raw-y (sklearn RF supports
                     criterion="poisson" since 1.0)

sklearn RF criterion does not support tweedie or gamma -- those raise in
__post_init__.

Forest as a whole is monotone in price_per_litre: each tree enforces the
constraint via the same monotone-aware splitter as DecisionTreeRegressor /
HistGradientBoosting; averaging trees preserves monotonicity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .preprocess import build_encoder
from ..config import CATEGORICAL_COLS

MONOTONIC_PRICE_FEAT = "price_per_litre"

_CRITERION_MAP = {
    "squared_error": "squared_error",
    "poisson":       "poisson",
}
_RAW_Y_OBJECTIVES = {"poisson"}


@dataclass
class RFModel:
    n_estimators: int = 800
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: float | str = 1.0
    max_samples: float | None = 0.8
    random_state: int = 42
    n_jobs: int = -1
    objective: str = "squared_error"
    feature_cols: list[str] | None = None

    def __post_init__(self):
        if self.objective not in _CRITERION_MAP:
            raise ValueError(
                f"objective must be one of {tuple(_CRITERION_MAP)}, got {self.objective!r}"
            )
        self.pipeline_: Pipeline | None = None
        self.est_: RandomForestRegressor | None = None
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
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=_CRITERION_MAP[self.objective],
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_samples=self.max_samples,
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            # monotonic_cst is set after prep.fit so we know the
            # post-encoding column order (sklearn RF expects list form).
        )
        return Pipeline([("prep", prep), ("model", model)])

    @staticmethod
    def _monotone_list(feature_names: list[str]) -> list[int]:
        return [-1 if f == MONOTONIC_PRICE_FEAT else 0 for f in feature_names]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RFModel":
        cols = self.feature_cols or list(X.columns)
        X = X[cols]
        self.pipeline_ = self._build(X)
        prep = self.pipeline_.named_steps["prep"]
        model = self.pipeline_.named_steps["model"]

        prep.fit(X, y)
        feat_names = list(prep.get_feature_names_out())
        model.set_params(monotonic_cst=self._monotone_list(feat_names))
        Xp = prep.transform(X)
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

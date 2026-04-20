"""Random Forest regressor on log target.

Categorical handling follows sklearn's ColumnTransformer pattern:
- low-cardinality (<= RF_HIGH_CARD_THRESHOLD levels) -> one-hot
- high-cardinality (>  RF_HIGH_CARD_THRESHOLD levels) -> smoothed target encoding
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .preprocess import LinearPreprocessor
from ..config import CATEGORICAL_COLS, RF_HIGH_CARD_THRESHOLD


@dataclass
class RFModel:
    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_leaf: int = 20
    max_features: str | float = "sqrt"
    random_state: int = 42
    n_jobs: int = -1
    feature_cols: list[str] | None = None

    def __post_init__(self):
        self.est_: RandomForestRegressor | None = None
        self._feature_order_: list[str] | None = None
        self._low_card_: list[str] = []
        self._high_card_: list[str] = []
        self._te_: LinearPreprocessor | None = None
        self._dummy_cols_: list[str] = []

    def _split_cats(self, X: pd.DataFrame) -> None:
        cats = [c for c in CATEGORICAL_COLS if c in X.columns]
        low, high = [], []
        for c in cats:
            if X[c].nunique(dropna=False) <= RF_HIGH_CARD_THRESHOLD:
                low.append(c)
            else:
                high.append(c)
        self._low_card_ = low
        self._high_card_ = high

    def _encode(self, X: pd.DataFrame, fit: bool, y: np.ndarray | None = None) -> pd.DataFrame:
        if self._high_card_:
            if fit:
                self._te_ = LinearPreprocessor(cat_cols=self._high_card_, num_cols=[])
                te_out = self._te_.fit_transform(X[self._high_card_], y)
            else:
                te_out = self._te_.transform(X[self._high_card_])
        else:
            te_out = pd.DataFrame(index=X.index)

        if self._low_card_:
            dummies = pd.get_dummies(
                X[self._low_card_], drop_first=True, dummy_na=False
            ).astype(float)
            if fit:
                self._dummy_cols_ = list(dummies.columns)
            else:
                dummies = dummies.reindex(columns=self._dummy_cols_, fill_value=0.0)
        else:
            dummies = pd.DataFrame(index=X.index)

        drop_cols = self._low_card_ + self._high_card_
        num = X.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])

        return pd.concat([num, te_out, dummies], axis=1)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RFModel":
        cols = self.feature_cols or list(X.columns)
        X_sub = X[cols]
        self._split_cats(X_sub)
        Xe = self._encode(X_sub, fit=True, y=y).fillna(0.0)
        self._feature_order_ = list(Xe.columns)
        self.est_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.est_.fit(Xe.values, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_cols or list(X.columns)
        Xe = self._encode(X[cols], fit=False).reindex(columns=self._feature_order_, fill_value=0.0)
        return self.est_.predict(Xe.fillna(0.0).values)

    @property
    def feature_importance_(self) -> pd.Series:
        return pd.Series(self.est_.feature_importances_, index=self._feature_order_).sort_values(ascending=False)

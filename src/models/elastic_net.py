"""Elastic Net / GLM regressor (Tier 1, interpretable baseline).

Pipeline: OHE (low-card cats) + TargetEncoder (high-card cats) + StandardScaler
(numerics) + a sklearn linear estimator selected by `objective`:

- `default`  → ElasticNet           (squared error + L1+L2 on log-y)
- `poisson`  → PoissonRegressor     (log-link GLM, L2 only, raw-y)
- `gamma`    → GammaRegressor       (log-link GLM, L2 only, raw-y)
- `tweedie`  → TweedieRegressor     (log-link, power=1.5, L2 only, raw-y)

GLM objectives trade L1 sparsity for an explicit distributional assumption
matching positive-integer volume data. Callers pass raw volume when
`expects_raw_y` is True, log-y otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet as SKElasticNet,
    GammaRegressor,
    PoissonRegressor,
    TweedieRegressor,
)
from sklearn.pipeline import Pipeline

from .preprocess import build_encoder
from ..config import CATEGORICAL_COLS, EN_HIGH_CARD_THRESHOLD

MONOTONIC_PRICE_FEAT = "log_price_per_litre"
_OBJECTIVES = ("default", "poisson", "tweedie", "gamma")
_TWEEDIE_POWER = 1.5


@dataclass
class ElasticNetModel:
    alpha: float = 1e-3
    l1_ratio: float = 0.5  # only used when objective="default"
    max_iter: int = 10_000
    random_state: int = 42
    objective: str = "default"
    feature_cols: list[str] | None = None

    def __post_init__(self):
        if self.objective not in _OBJECTIVES:
            raise ValueError(
                f"objective must be one of {_OBJECTIVES}, got {self.objective!r}"
            )
        self.pipeline_: Pipeline | None = None
        self._feature_order_: list[str] | None = None

    @property
    def expects_raw_y(self) -> bool:
        return self.objective != "default"

    def _split_cols(self, cols: list[str]) -> tuple[list[str], list[str]]:
        cats = [c for c in CATEGORICAL_COLS if c in cols]
        nums = [c for c in cols if c not in cats]
        return nums, cats

    def _build_estimator(self):
        if self.objective == "default":
            return SKElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        kw = dict(alpha=self.alpha, max_iter=self.max_iter)
        if self.objective == "poisson":
            return PoissonRegressor(**kw)
        if self.objective == "gamma":
            return GammaRegressor(**kw)
        if self.objective == "tweedie":
            return TweedieRegressor(power=_TWEEDIE_POWER, **kw)
        raise ValueError(f"unsupported objective {self.objective}")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ElasticNetModel":
        cols = self.feature_cols or list(X.columns)
        X = X[cols]
        nums, cats = self._split_cols(cols)
        prep = build_encoder(
            X, cat_cols=cats, num_cols=nums,
            high_card_threshold=EN_HIGH_CARD_THRESHOLD,
            scale_numeric=True,
        )
        model = self._build_estimator()
        self.pipeline_ = Pipeline([("prep", prep), ("model", model)])
        self.pipeline_.fit(X, y)
        self._feature_order_ = list(
            self.pipeline_.named_steps["prep"].get_feature_names_out()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_cols or list(X.columns)
        return self.pipeline_.predict(X[cols])

    @property
    def coefficients(self) -> pd.Series:
        est = self.pipeline_.named_steps["model"]
        return pd.Series(est.coef_, index=self._feature_order_).sort_values()

    def own_price_elasticity(self) -> float | None:
        """Recover interpretable elasticity for `log_price_per_litre`.

        For `default` (OLS on log-y): β is standardized, un-standardize via
        the scaler std to get %volume/%price.

        For GLM (Poisson/Gamma/Tweedie with log-link on raw y): β is also
        standardized; since x = log_price is itself logged, un-standardized
        β is d log(E[y]) / d log(price) = elasticity.

        Returns None if log_price_per_litre isn't in the pipeline. Sign is
        NOT guaranteed negative — caller should sign-test.
        """
        feat = MONOTONIC_PRICE_FEAT
        if feat not in self._feature_order_:
            return None
        est = self.pipeline_.named_steps["model"]
        prep = self.pipeline_.named_steps["prep"]
        idx = self._feature_order_.index(feat)
        beta_std = float(est.coef_[idx])

        scaler = prep.named_transformers_.get("num")
        if scaler is None or isinstance(scaler, str):
            return beta_std
        scaler_cols = list(scaler.feature_names_in_)
        if feat not in scaler_cols:
            return beta_std
        std = float(scaler.scale_[scaler_cols.index(feat)])
        return beta_std / std if std > 0 else beta_std

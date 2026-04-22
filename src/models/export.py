"""Export champion as a self-contained sklearn object.

The saved joblib contains only sklearn + xgboost + lightgbm classes, so
downstream apps can `joblib.load(path).predict(engineered_df)` with no
custom imports.

Two code paths depending on the wrapper's training y-space:

- log-y objective (default / ElasticNet MSE):
  clone the inner Pipeline, wrap in `TransformedTargetRegressor(log1p, expm1)`,
  fit on raw volume so `.predict()` returns raw.

- raw-y objective (Poisson / Tweedie / Gamma / XGB squaredlogerror):
  the estimator already handles raw y via log-link / internal log1p; clone
  the inner Pipeline, fit directly on raw volume, skip TTR.

In both cases the downstream contract is identical:
`joblib.load(path).predict(engineered_df)` → raw nielsen_total_volume.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor


def export_champion(
    wrapper,
    X: pd.DataFrame,
    y_raw: np.ndarray,
    output_path: str | Path,
):
    """Re-fit a structural copy of the wrapper's Pipeline on raw volume and
    dump to joblib. Returns the fitted exported object (Pipeline or TTR)."""
    if getattr(wrapper, "pipeline_", None) is None:
        raise ValueError(
            "wrapper.pipeline_ is None — call wrapper.fit() before export."
        )

    inner = clone(wrapper.pipeline_)
    if getattr(wrapper, "expects_raw_y", False):
        final = inner
        final.fit(X, y_raw)
    else:
        final = TransformedTargetRegressor(
            regressor=inner,
            func=np.log1p,
            inverse_func=np.expm1,
        )
        final.fit(X, y_raw)
    joblib.dump(final, str(output_path))
    return final

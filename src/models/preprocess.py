"""Shared preprocessing for all 4 models.

Public entry point: `build_encoder(...)` returns a sklearn ColumnTransformer
that dispatches low-cardinality cats to OneHotEncoder and high-cardinality
cats to TargetEncoder, with optional StandardScaler on numerics.

Use inside a sklearn Pipeline so the saved model is a pure sklearn object
(no custom classes in the pickled artifact).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder


def build_encoder(
    X_sample: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    high_card_threshold: int = 20,
    scale_numeric: bool = False,
    te_smooth: float = 20.0,
) -> ColumnTransformer:
    """Build a ColumnTransformer that splits cats by cardinality.

    - cats with nunique <= high_card_threshold -> OneHotEncoder(drop="first")
    - cats with nunique >  high_card_threshold -> TargetEncoder(smooth=20)
    - nums passthrough by default; scale_numeric=True adds StandardScaler
      (needed for linear models, not for trees).

    `verbose_feature_names_out=False` keeps numeric column names unchanged
    (e.g. "price_per_litre" stays as-is through passthrough), so tree
    models can anchor monotone constraints to stable names.
    """
    cat_cols = [c for c in cat_cols if c in X_sample.columns]
    num_cols = [c for c in num_cols if c in X_sample.columns]

    low_card = [c for c in cat_cols
                if X_sample[c].nunique(dropna=False) <= high_card_threshold]
    high_card = [c for c in cat_cols if c not in low_card]

    transformers = []
    if num_cols:
        transformers.append(
            ("num", StandardScaler() if scale_numeric else "passthrough", num_cols)
        )
    if low_card:
        transformers.append((
            "cat_low",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            low_card,
        ))
    if high_card:
        transformers.append((
            "cat_high",
            TargetEncoder(target_type="continuous", smooth=te_smooth),
            high_card,
        ))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Emit DataFrames so downstream estimators (XGB/LGB/HGB) can resolve
    # column-name-keyed params like monotone_constraints={"price_per_litre": -1}.
    ct.set_output(transform="pandas")
    return ct

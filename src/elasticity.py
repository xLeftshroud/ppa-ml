"""Elasticity extraction + plausibility + stability checks.

Five model-specific extractors return a uniform DataFrame:
    columns = [sku, customer, beta_mean, beta_lo, beta_hi, beta_std, method]

Plausibility checks:
    - sign_test:       share of SKUs with beta < 0 (target > 95%)
    - magnitude_test:  share of SKUs with beta in [-3.5, -0.5]
    - stability:       coefficient of variation across bootstraps (target < 0.3)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SOFT_DRINK_LOWER = -3.5
SOFT_DRINK_UPPER = -0.5


# ------------------------------------------------------------------
# Per-model extractors
# ------------------------------------------------------------------
def elastic_net_elasticity(model, df_panel: pd.DataFrame) -> pd.DataFrame:
    """Elastic Net is log-log: beta on log_price_per_litre IS elasticity.

    Single global elasticity -> broadcast to each (sku, customer) pair.
    """
    beta = model.own_price_elasticity()
    if beta is None:
        beta = np.nan
    panel = df_panel[["product_sku_code", "customer"]].drop_duplicates().reset_index(drop=True)
    panel["beta_mean"] = beta
    panel["beta_lo"] = np.nan
    panel["beta_hi"] = np.nan
    panel["beta_std"] = np.nan
    panel["method"] = "log_log_coef"
    return panel


def tree_local_elasticity(
    model,
    df_panel: pd.DataFrame,
    feature_cols: list[str],
    price_col: str | None = None,
    delta: float = 0.10,
    predict_is_raw: bool = False,
) -> pd.DataFrame:
    """Numerical local elasticity: (d log_vol / d log_price) at each row,
    aggregated to (sku, customer) by median. Auto-detects whether the
    price feature is on log scale or raw scale and perturbs accordingly.

    Uses a **symmetric central difference** with delta=0.10 (±10% price
    perturbation). Trees are piecewise-constant, so a tiny one-sided step
    often lands inside a flat leaf and gives β=0 even when the monotone
    constraint is active. ±10% is large enough to cross most split
    boundaries while still being "local" for elasticity interpretation.

    `predict_is_raw`: set True when the model returns raw volume (GLM /
    exported joblib wrapped in TTR); the function log-transforms predictions
    before the finite-difference."""
    if price_col is None or price_col not in feature_cols:
        if "log_price_per_litre" in feature_cols:
            price_col = "log_price_per_litre"
        elif "price_per_litre" in feature_cols:
            price_col = "price_per_litre"
        else:
            raise ValueError(
                "feature_cols must contain 'log_price_per_litre' or 'price_per_litre'."
            )
    log_price = price_col.startswith("log_")

    X = df_panel[feature_cols].copy()
    X_up = X.copy()
    X_dn = X.copy()
    if log_price:
        X_up[price_col] = X_up[price_col] + delta
        X_dn[price_col] = X_dn[price_col] - delta
        denom = 2.0 * delta
    else:
        X_up[price_col] = X_up[price_col] * (1.0 + delta)
        X_dn[price_col] = X_dn[price_col] * (1.0 - delta)
        denom = float(np.log(1.0 + delta) - np.log(1.0 - delta))
    y_up = model.predict(X_up)
    y_dn = model.predict(X_dn)

    if predict_is_raw:
        y_up = np.log1p(np.clip(y_up, 0, None))
        y_dn = np.log1p(np.clip(y_dn, 0, None))
    local_elast = (y_up - y_dn) / denom
    tmp = pd.DataFrame(
        {
            "product_sku_code": df_panel["product_sku_code"].values,
            "customer": df_panel["customer"].values,
            "elast": local_elast,
        }
    )
    agg = (
        tmp.groupby(["product_sku_code", "customer"])["elast"]
        .agg(["median", "std", lambda s: s.quantile(0.025), lambda s: s.quantile(0.975)])
        .reset_index()
    )
    agg.columns = ["product_sku_code", "customer", "beta_mean", "beta_std", "beta_lo", "beta_hi"]
    agg["method"] = "numerical_local"
    return agg


def bayesian_elasticity(
    model, df_panel: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Extract per-cell posterior and broadcast to each SKU in that cell.

    v6 model posts elasticity at the (top_brand, flavor_internal, pack_tier) cell
    level. SKUs in the same cell share a beta. If ``df_panel`` is None, returns
    the raw per-cell table (one row per cell).
    """
    post = model.elasticity_posterior()  # per-cell
    post = post.rename(
        columns={"beta_hdi_low": "beta_lo", "beta_hdi_high": "beta_hi"}
    )
    if df_panel is None:
        out = post.copy()
        out["customer"] = "ALL"
        out["method"] = "bayesian_hier"
        return out[[
            "top_brand", "flavor_internal", "pack_tier", "n_skus_in_cell",
            "customer", "beta_mean", "beta_lo", "beta_hi", "beta_std", "method",
        ]]
    keys = ["top_brand", "flavor_internal", "pack_tier"]
    panel = df_panel[["product_sku_code", "customer"] + keys].drop_duplicates()
    merged = panel.merge(post[keys + ["beta_mean", "beta_lo", "beta_hi", "beta_std"]],
                         on=keys, how="left")
    merged["method"] = "bayesian_hier"
    return merged[[
        "product_sku_code", "customer", "beta_mean", "beta_lo", "beta_hi",
        "beta_std", "method",
    ]]


# ------------------------------------------------------------------
# Plausibility tests
# ------------------------------------------------------------------
def sign_test(elast_df: pd.DataFrame, col: str = "beta_mean") -> dict:
    """Share of SKUs with beta<0. Also breaks out the zero bucket, because
    tree elasticities often land exactly at 0 in feature regions where the
    model never split on price; those are "no signal" rather than "wrong
    direction" and shouldn't be conflated with positive betas."""
    valid = elast_df[col].dropna()
    n = len(valid)
    share_neg = float((valid < 0).mean()) if n else np.nan
    share_zero = float((valid == 0).mean()) if n else np.nan
    nonzero = valid[valid != 0]
    share_neg_nonzero = float((nonzero < 0).mean()) if len(nonzero) else np.nan
    return {
        "share_negative": share_neg,
        "share_zero": share_zero,
        "share_negative_among_nonzero": share_neg_nonzero,
        "n": int(n),
        "n_nonzero": int(len(nonzero)),
        "passes_95pct": share_neg > 0.95,
        "passes_95pct_nonzero": share_neg_nonzero > 0.95 if not np.isnan(share_neg_nonzero) else False,
    }


def magnitude_test(
    elast_df: pd.DataFrame,
    col: str = "beta_mean",
    lo: float = SOFT_DRINK_LOWER,
    hi: float = SOFT_DRINK_UPPER,
) -> dict:
    valid = elast_df[col].dropna()
    share_in_range = float(((valid >= lo) & (valid <= hi)).mean()) if len(valid) else np.nan
    return {
        "share_in_range": share_in_range,
        "median": float(valid.median()) if len(valid) else np.nan,
        "industry_range": (lo, hi),
    }


def stability_cv(
    bootstrap_betas: np.ndarray,
) -> dict:
    """Given an (n_boot, n_units) matrix of betas, compute CV per unit."""
    mean = bootstrap_betas.mean(axis=0)
    std = bootstrap_betas.std(axis=0, ddof=1)
    cv = np.where(np.abs(mean) > 1e-6, std / np.abs(mean), np.nan)
    sign_flip = (np.sign(bootstrap_betas) != np.sign(mean)).mean(axis=0)
    return {
        "cv_per_unit": cv,
        "sign_flip_rate_per_unit": sign_flip,
        "median_cv": float(np.nanmedian(cv)),
        "max_sign_flip": float(np.nanmax(sign_flip)) if sign_flip.size else np.nan,
    }


def plausibility_scorecard(elast_df: pd.DataFrame) -> pd.Series:
    """One-row summary for the scorecard table.

    Reports both the strict `share_negative` and the nonzero-aware
    `share_negative_among_nonzero`. For monotone-constrained trees the
    latter is the honest sign check because β=0 means "tree didn't split
    on price in that SKU's region" (no signal) rather than "tree learned
    wrong direction"."""
    st = sign_test(elast_df)
    mt = magnitude_test(elast_df)
    return pd.Series(
        {
            "share_negative_beta": st["share_negative"],
            "share_zero_beta": st["share_zero"],
            "share_negative_among_nonzero": st["share_negative_among_nonzero"],
            "sign_test_pass": st["passes_95pct"],
            "sign_test_pass_nonzero": st["passes_95pct_nonzero"],
            "median_beta": mt["median"],
            "share_in_soft_drink_range": mt["share_in_range"],
        }
    )

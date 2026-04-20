"""Feature engineering for PPA: 7 new features + target log-transform.

Applied AFTER time-split to prevent leakage (group-wise stats computed
on training portion only, then applied to validation/test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-free features (computed per row, no group stats).

    Safe to apply before train/val split.
    """
    out = df.copy()

    out["price_imputed_flag"] = out["_price_before_impute"].isna().astype(int)
    out["total_pack_volume_ml"] = (
        out["pack_size_internal"] * out["units_per_package_internal"]
    )
    out["pack_tier"] = _assign_pack_tier(out)
    out["log_volume"] = np.log1p(out["nielsen_total_volume"])
    out["log_price_per_litre"] = np.log(out["price_per_litre"].clip(lower=1e-6))

    return out


def add_panel_features(
    train_df: pd.DataFrame, apply_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Leakage-prone features: group statistics & lags.

    Stats are fitted on ``train_df`` and applied to ``apply_df`` (defaults
    to the same frame so it also works for single-frame EDA). For CV/test
    evaluation, always pass train + holdout separately.
    """
    apply_df = train_df if apply_df is None else apply_df

    # --- brand median price-per-litre (fit on train) -----------------
    brand_median = (
        train_df.groupby("top_brand")["price_per_litre"].median().rename("_brand_med")
    )
    tier_median = (
        train_df.groupby(["pack_type_internal", "pack_tier"])["price_per_litre"]
        .median()
        .rename("_tier_med")
    )

    out = apply_df.merge(brand_median, left_on="top_brand", right_index=True, how="left")
    out = out.merge(
        tier_median,
        left_on=["pack_type_internal", "pack_tier"],
        right_index=True,
        how="left",
    )
    # fallback to global median for unseen groups in holdout
    global_med = float(train_df["price_per_litre"].median())
    out["_brand_med"] = out["_brand_med"].fillna(global_med)
    out["_tier_med"] = out["_tier_med"].fillna(global_med)

    out["price_premium_vs_brand"] = out["price_per_litre"] / out["_brand_med"]
    out["price_premium_vs_pack_tier"] = out["price_per_litre"] / out["_tier_med"]
    out = out.drop(columns=["_brand_med", "_tier_med"])

    # --- promo depth: 1 - current / rolling 12w max ------------------
    sort_time_col = "continuous_week" if "continuous_week" in out.columns else "yearweek"
    out = out.sort_values(["product_sku_code", "customer", sort_time_col]).reset_index(
        drop=True
    )
    grp = out.groupby(["product_sku_code", "customer"], group_keys=False)
    out["_rolling_max_price"] = grp["price_per_litre"].transform(
        lambda s: s.rolling(window=12, min_periods=2).max().shift(1)
    )
    out["promo_depth"] = 1.0 - out["price_per_litre"] / out["_rolling_max_price"]
    out["promo_depth"] = out["promo_depth"].fillna(0.0).clip(lower=-0.5, upper=0.8)
    out = out.drop(columns=["_rolling_max_price"])

    # --- lag features (within panel unit) ----------------------------
    out["log_volume_lag1"] = grp["log_volume"].shift(1)
    out["log_volume_lag4"] = grp["log_volume"].shift(4)
    # within-panel bfill so all models see identical non-NaN inputs
    out["log_volume_lag1"] = grp["log_volume"].shift(1).bfill()
    out["log_volume_lag4"] = grp["log_volume"].shift(4).bfill()

    return out


def _assign_pack_tier(df: pd.DataFrame) -> pd.Series:
    """single-serve / multi-pack-take-home / large-format / other."""
    ps = df["pack_size_internal"]
    upk = df["units_per_package_internal"]
    tvol = ps * upk

    single = (ps < 500) & (upk == 1)
    multi_home = upk >= 6
    large_fmt = tvol >= 1500

    tier = pd.Series("other", index=df.index, dtype="object")
    tier[large_fmt] = "large_format"
    tier[multi_home & ~large_fmt] = "multi_pack_take_home"
    tier[single] = "single_serve"
    return tier


def build_features(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Full feature pipeline. Returns (train_feat, holdout_feat)."""
    train_b = add_base_features(train_df)
    train_out = add_panel_features(train_b, train_b)

    holdout_out = None
    if holdout_df is not None:
        hold_b = add_base_features(holdout_df)
        holdout_out = add_panel_features(train_b, hold_b)

    return train_out, holdout_out

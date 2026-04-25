"""Row-wise, cold-start-safe feature transforms for PPA.

All features in this module are pure row-wise functions of product
attributes — they do NOT depend on panel history (no lag, no rolling,
no per-panel group statistics). This keeps every feature well-defined
for unseen (sku x customer) combinations used in scenario simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_per_litre(df: pd.DataFrame) -> pd.Series:
    """Price per litre = price_per_item / total_pack_volume_litres."""
    volume_l = (df["pack_size_internal"] * df["units_per_package_internal"]) / 1000.0
    return df["price_per_item"] / volume_l


def price_per_100ml(df: pd.DataFrame) -> pd.Series:
    """Price per 100ml. Perfectly collinear with price_per_litre (factor 10).

    Kept for reporting; excluded from CANDIDATE_FEATURES for modelling.
    """
    volume_100ml = (df["pack_size_internal"] * df["units_per_package_internal"]) / 100.0
    return df["price_per_item"] / volume_100ml


def week(df: pd.DataFrame) -> pd.Series:
    """Week-of-year extracted from yearweek (YYYYWW)."""
    return df["yearweek"] % 100


def week_sin(df: pd.DataFrame) -> pd.Series:
    """Cyclical sin encoding of week-of-year."""
    return np.sin(2 * np.pi * df["week"] / 52)


def week_cos(df: pd.DataFrame) -> pd.Series:
    """Cyclical cos encoding of week-of-year."""
    return np.cos(2 * np.pi * df["week"] / 52)


def continuous_week(df: pd.DataFrame) -> pd.Series:
    """Zero-based trend index: rank of yearweek among distinct yearweeks in df."""
    return df["yearweek"].rank(method="dense").astype(int) - 1


def pack_size_total(df: pd.DataFrame) -> pd.Series:
    """Total milliliters per package = single-unit size x units per pack."""
    return df["pack_size_internal"] * df["units_per_package_internal"]


def pack_tier(df: pd.DataFrame) -> pd.Series:
    """Coarse pack format: single_serve / multi_pack_take_home / large_format / other."""
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


def log_nielsen_total_volume(df: pd.DataFrame) -> pd.Series:
    """log1p of raw packs sold. Kept for EDA / legacy comparisons; not the
    training target (see log_volume_in_litres)."""
    return np.log1p(df["nielsen_total_volume"])


def volume_in_litres(df: pd.DataFrame) -> pd.Series:
    """Liquid volume sold (litres) = packs * units_per_pack * pack_size_ml / 1000.

    nielsen_total_volume is the count of packs sold; multiplying by
    units_per_package_internal and pack_size_internal/1000 converts to
    actual liquid litres dispensed, the cross-pack-comparable target.
    """
    return (
        df["nielsen_total_volume"]
        * df["units_per_package_internal"]
        * df["pack_size_internal"]
        / 1000.0
    )


def log_volume_in_litres(df: pd.DataFrame) -> pd.Series:
    """log1p of liquid volume sold (training target)."""
    return np.log1p(df["volume_in_litres"].clip(lower=0))


def log_price_per_litre(df: pd.DataFrame) -> pd.Series:
    """Natural log of price_per_litre with a floor to avoid log(0)."""
    return np.log(df["price_per_litre"].clip(lower=1e-6))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transforms. Returns a new frame (does not mutate input).

    Existing columns of the same name are overwritten, so this is idempotent
    on the shipped dataset/dataset_cleaned.csv and also correct when fed the
    origin cleaned dataset (before notebook preprocessing).
    """
    out = df.copy()
    out["price_per_litre"] = price_per_litre(out)
    out["price_per_100ml"] = price_per_100ml(out)
    out["week"] = week(out)
    out["week_sin"] = week_sin(out)
    out["week_cos"] = week_cos(out)
    out["continuous_week"] = continuous_week(out)
    out["pack_size_total"] = pack_size_total(out)
    out["pack_tier"] = pack_tier(out)
    out["log_nielsen_total_volume"] = log_nielsen_total_volume(out)
    out["volume_in_litres"] = volume_in_litres(out)
    out["log_volume_in_litres"] = log_volume_in_litres(out)
    out["log_price_per_litre"] = log_price_per_litre(out)
    return out

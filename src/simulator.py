"""What-if scenario engine.

Wraps any fitted model into a clean ``simulate(sku, customer, dp_pct, promo_on)``
call that returns baseline vs scenario volume + lift %. Also exposes
sanity_check() which scans a grid of price shocks and flags any non-physical
behavior (positive own-price response, or negative predicted volume).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Simulator:
    model: object            # anything with .predict(DataFrame) -> log_nielsen_total_volume
    feature_cols: list[str]
    df_reference: pd.DataFrame

    # optional training distribution for guard rails
    price_col: str = "price_per_litre"
    log_price_col: str = "log_price_per_litre"
    margin: float | None = None   # reserved; profit-opt not supported yet

    def __post_init__(self):
        p = self.df_reference[self.price_col].dropna()
        self._price_p5 = float(p.quantile(0.05))
        self._price_p95 = float(p.quantile(0.95))

    def _baseline_row(self, sku, customer) -> pd.Series:
        mask = (
            (self.df_reference["product_sku_code"] == sku)
            & (self.df_reference["customer"] == customer)
        )
        sub = self.df_reference[mask]
        if sub.empty:
            raise ValueError(f"No reference row for sku={sku}, customer={customer}")
        return sub.sort_values("yearweek").iloc[-1].copy()

    def simulate(
        self,
        sku,
        customer,
        price_change_pct: float = 0.0,
        promo_on: int | None = None,
    ) -> dict:
        base = self._baseline_row(sku, customer)
        scen = base.copy()
        new_price = base[self.price_col] * (1 + price_change_pct)
        scen[self.price_col] = new_price
        scen[self.log_price_col] = np.log(max(new_price, 1e-6))
        if promo_on is not None:
            scen["promotion_indicator"] = int(promo_on)

        X_base = pd.DataFrame([base[self.feature_cols]])
        X_scen = pd.DataFrame([scen[self.feature_cols]])

        vol_base = float(np.expm1(self.model.predict(X_base)[0]))
        vol_scen = float(np.expm1(self.model.predict(X_scen)[0]))
        vol_scen = max(vol_scen, 0.0)
        lift_pct = (vol_scen - vol_base) / vol_base if vol_base > 0 else np.nan

        return {
            "sku": sku,
            "customer": customer,
            "price_change_pct": price_change_pct,
            "promo_on": int(promo_on) if promo_on is not None else int(base["promotion_indicator"]),
            "baseline_price": float(base[self.price_col]),
            "scenario_price": float(new_price),
            "baseline_volume": vol_base,
            "scenario_volume": vol_scen,
            "volume_lift_pct": lift_pct,
            "price_in_training_range": self._price_p5 <= new_price <= self._price_p95,
        }

    def sanity_check(
        self,
        sku,
        customer,
        shocks: tuple[float, ...] = (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2),
    ) -> pd.DataFrame:
        rows = [self.simulate(sku, customer, price_change_pct=s) for s in shocks]
        df = pd.DataFrame(rows)
        df["sign_ok"] = np.sign(-df["price_change_pct"]) == np.sign(df["volume_lift_pct"])
        df["volume_nonneg"] = df["scenario_volume"] >= 0
        return df

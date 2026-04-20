"""Bayesian hierarchical track. Runs once per prior configuration.

Usage:
    python -m scripts.run_bayesian --prior moderate --draws 2000

This is the Tier-3 differentiator: partial-pooled SKU-level elasticities
with 95% HDI. Expect 4-6 hours on CPU for 923 panel units x 2 chains.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PATH, OUTPUTS
from src.features import add_base_features, add_panel_features
from src.split import expanding_window_cv, final_holdout_split
from src.models.hier_bayes import BayesianHierModel
from src.elasticity import bayesian_elasticity
from src.evaluate import metrics_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["weak", "moderate", "strong"], default="moderate")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--fold", type=int, default=-1, help="which CV fold to use for train (-1 = all dev)")
    args = ap.parse_args()

    df = pd.read_csv(DATA_PATH)
    df_b = add_base_features(df)
    df_fe = add_panel_features(df_b, df_b)
    df_fe = df_fe.dropna(subset=["log_volume"]).reset_index(drop=True)

    dev_idx, test_idx = final_holdout_split(df_fe)
    df_dev = df_fe.iloc[dev_idx].reset_index(drop=True)
    df_test = df_fe.iloc[test_idx].reset_index(drop=True)

    if args.fold >= 0:
        folds = expanding_window_cv(df_dev)
        tr_idx, va_idx = folds[args.fold]
        df_train = df_dev.iloc[tr_idx].reset_index(drop=True)
        df_val = df_dev.iloc[va_idx].reset_index(drop=True)
    else:
        df_train = df_dev
        df_val = df_test if len(df_test) else None

    print(f"Bayesian track: prior={args.prior}, draws={args.draws}, chains={args.chains}")
    print(f"  train rows={len(df_train)}, val rows={len(df_val) if df_val is not None else 0}")

    model = BayesianHierModel(
        prior_scale=args.prior,
        num_warmup=args.warmup,
        num_samples=args.draws,
        num_chains=args.chains,
    )
    t0 = time.perf_counter()
    model.fit(df_train, df_train["log_volume"].values)
    dur = time.perf_counter() - t0
    print(f"  fit wall-clock: {dur/60:.1f} min")

    elast = bayesian_elasticity(model, df_dev)
    suffix = f"_{args.prior}"
    elast.to_csv(OUTPUTS / f"elasticity_hier_bayes{suffix}.csv", index=False)

    if df_val is not None and len(df_val):
        val_pred = model.predict(df_val)
        m = metrics_table(df_val["log_volume"].values, val_pred, train_time_sec=dur)
        (OUTPUTS / f"metrics_hier_bayes{suffix}.json").write_text(json.dumps(m, indent=2))
        print(f"  val metrics: {m}")

    import numpyro
    import arviz as az

    nc_path = OUTPUTS / f"model_hier_bayes{suffix}.nc"
    meta_path = OUTPUTS / f"metadata_hier_bayes{suffix}.json"
    model.idata_.to_netcdf(nc_path)

    metadata = {
        "model_type": "hier_bayes",
        "prior_scale": args.prior,
        "num_warmup": int(args.warmup),
        "num_samples": int(args.draws),
        "num_chains": int(args.chains),
        "brand_codes": model._brand_codes_,
        "sku_codes": model._sku_codes_,
        "feature_cols": [
            "log_price_per_litre", "promotion_indicator", "week_sin", "week_cos",
        ],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": sys.version.split()[0],
            "numpyro": numpyro.__version__,
            "arviz": az.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"  saved {nc_path} + {meta_path}")
    print("done.")


if __name__ == "__main__":
    main()

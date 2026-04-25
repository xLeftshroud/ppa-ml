"""Bayesian hierarchical track (v6: brand-nested + flavor/pack crossed).

Usage:
    python -m scripts.run_bayesian --prior moderate --draws 2000

Tier-3 differentiator: per-cell (brand x flavor x pack_tier) elasticity with
95% HDI, plus full MCMC convergence diagnostics. Expect 45-90 min on CPU for
~100-150 cells x 2 chains.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import numpyro
# Must be set before any jax/numpyro computation; enables true parallel chains on CPU.
numpyro.set_host_device_count(2)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PATH, OUTPUTS, SEEDS, TARGET
from src.features import build_features
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
    ap.add_argument("--cv", action="store_true",
                    help="Run |SEEDS| x N_SPLITS MCMC fits and write metrics_hier_bayes_<prior>.csv; "
                         "final refit on full dev still runs afterward.")
    args = ap.parse_args()

    df = pd.read_csv(DATA_PATH)
    df_fe = build_features(df)
    df_fe = df_fe.dropna(subset=[TARGET]).reset_index(drop=True)

    dev_idx, test_idx = final_holdout_split(df_fe)
    df_dev = df_fe.iloc[dev_idx].reset_index(drop=True)
    df_test = df_fe.iloc[test_idx].reset_index(drop=True)

    suffix = f"_{args.prior}"

    # ------------------------------------------------------------------
    # Optional CV loop: |SEEDS| x N_SPLITS MCMC fits -> metrics_hier_bayes_<prior>.csv
    # Matches GBM `metrics_<model>.csv` schema so compare_runs.py L1/L2
    # leaderboards include Bayesian.
    # ------------------------------------------------------------------
    if args.cv:
        folds = expanding_window_cv(df_dev)
        print(f"[CV] {len(SEEDS)} seeds x {len(folds)} folds = {len(SEEDS) * len(folds)} MCMC fits")
        cv_rows = []
        cv_start = time.perf_counter()
        for seed in SEEDS:
            for fi, (tr_idx, va_idx) in enumerate(folds, start=1):
                df_tr = df_dev.iloc[tr_idx].reset_index(drop=True)
                df_va = df_dev.iloc[va_idx].reset_index(drop=True)
                print(f"[CV] seed={seed} fold={fi}: train={len(df_tr)} val={len(df_va)}")
                m_cv = BayesianHierModel(
                    prior_scale=args.prior,
                    num_warmup=args.warmup,
                    num_samples=args.draws,
                    num_chains=args.chains,
                    random_state=seed,
                )
                t0 = time.perf_counter()
                m_cv.fit(df_tr, df_tr[TARGET].values)
                dt = time.perf_counter() - t0
                val_pred = m_cv.predict(df_va)
                m = metrics_table(
                    df_va[TARGET].values, val_pred, train_time_sec=dt,
                )
                m.update({"model": "hier_bayes", "seed": seed, "fold": fi})
                cv_rows.append(m)
                print(f"       RMSE={m['rmse']:.0f}  WMAPE={m['wmape']:.4f}  time={dt/60:.1f} min")
                # release JAX compiled caches so memory does not balloon across fits
                try:
                    import jax
                    jax.clear_caches()
                except Exception:
                    pass
                del m_cv

        cv_df = pd.DataFrame(cv_rows)
        cv_path = OUTPUTS / f"metrics_hier_bayes{suffix}.csv"
        cv_df.to_csv(cv_path, index=False)
        print(f"[CV] saved {cv_path} ({len(cv_df)} rows, total {(time.perf_counter()-cv_start)/60:.1f} min)")

    if args.fold >= 0:
        folds = expanding_window_cv(df_dev)
        tr_idx, va_idx = folds[args.fold]
        df_train = df_dev.iloc[tr_idx].reset_index(drop=True)
        df_val = df_dev.iloc[va_idx].reset_index(drop=True)
    else:
        df_train = df_dev
        df_val = df_test if len(df_test) else None

    print(f"Bayesian v6: prior={args.prior}, draws={args.draws}, chains={args.chains}")
    print(f"  train rows={len(df_train)}, val rows={len(df_val) if df_val is not None else 0}")

    model = BayesianHierModel(
        prior_scale=args.prior,
        num_warmup=args.warmup,
        num_samples=args.draws,
        num_chains=args.chains,
    )
    t0 = time.perf_counter()
    model.fit(df_train, df_train[TARGET].values)
    dur = time.perf_counter() - t0
    print(f"  fit wall-clock: {dur/60:.1f} min")
    print(f"  n_brand={len(model._brand_codes_)} "
          f"n_flavor={len(model._flavor_codes_)} "
          f"n_pack={len(model._pack_codes_)} "
          f"n_cell={len(model._cell_codes_)}")

    # --- convergence diagnostics ---
    import arviz as az
    summary = model.convergence_summary()
    conv_path = OUTPUTS / f"convergence_hier_bayes{suffix}.csv"
    summary.to_csv(conv_path)
    n_div = model.divergences()
    rhat_max = float(summary["r_hat"].max()) if "r_hat" in summary.columns else float("nan")
    ess_min = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else float("nan")
    print(f"  rhat max={rhat_max:.4f}, ess_bulk min={ess_min:.0f}, divergences={n_div}")
    if rhat_max >= 1.01 or n_div > 0:
        print(f"  [WARN] convergence targets not met (rhat<1.01, divergences==0)")

    # --- per-cell elasticity posterior ---
    elast = bayesian_elasticity(model, df_dev)
    elast.to_csv(OUTPUTS / f"elasticity_hier_bayes{suffix}.csv", index=False)
    # also save the raw per-cell posterior (no customer broadcast) for inspection
    model.elasticity_posterior().to_csv(
        OUTPUTS / f"elasticity_cells_hier_bayes{suffix}.csv", index=False
    )

    if df_val is not None and len(df_val):
        val_pred = model.predict(df_val)
        m = metrics_table(df_val[TARGET].values, val_pred, train_time_sec=dur)
        (OUTPUTS / f"test_metrics_hier_bayes{suffix}.json").write_text(json.dumps(m, indent=2))
        print(f"  val metrics: {m}")

    nc_path = OUTPUTS / f"model_hier_bayes{suffix}.nc"
    meta_path = OUTPUTS / f"metadata_hier_bayes{suffix}.json"
    model.idata_.to_netcdf(nc_path)

    metadata = {
        "model_type": "hier_bayes_v6",
        "prior_scale": args.prior,
        "num_warmup": int(args.warmup),
        "num_samples": int(args.draws),
        "num_chains": int(args.chains),
        "brand_codes": model._brand_codes_,
        "flavor_codes": model._flavor_codes_,
        "pack_codes": model._pack_codes_,
        "cell_keys": [list(k) for k in model._cell_keys_],
        "customer_levels": model._customer_levels_,
        "feature_cols": [
            "log_price_per_litre", "promotion_indicator", "week_sin", "week_cos",
            "customer",
        ],
        "convergence": {
            "rhat_max": rhat_max,
            "ess_bulk_min": ess_min,
            "divergences": n_div,
        },
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": sys.version.split()[0],
            "numpyro": numpyro.__version__,
            "arviz": az.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"  saved {nc_path} + {meta_path} + {conv_path}")
    print("done.")


if __name__ == "__main__":
    main()

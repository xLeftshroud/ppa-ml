"""Run only the naive + seasonal_naive baselines, no ML training.

Produces ``outputs/metrics_naive.csv`` and ``outputs/metrics_seasonal_naive.csv``
with the same SEEDS x N_SPLITS schema that ``compare_runs.py`` expects.
Skips if the file already exists, matching ``run_model.py:[5/6] Baselines``
behaviour. Delete the file if you want a clean regeneration.

Usage:
    python -m scripts.run_baselines
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PATH, OUTPUTS, SEEDS, TARGET
from src.features import build_features
from src.split import final_holdout_split
from src.baselines import naive_predict, seasonal_naive_predict
from src.experiments import run_baseline_across_seeds


def main() -> None:
    print("[1/2] Loading data + engineering features...")
    df_fe = build_features(pd.read_csv(DATA_PATH))
    dev_idx, _ = final_holdout_split(df_fe)
    df_dev = df_fe.iloc[dev_idx].reset_index(drop=True)
    df_dev = df_dev.dropna(subset=["nielsen_total_volume"]).reset_index(drop=True)
    y_dev = df_dev[TARGET].values

    print(f"[2/2] Baselines on {len(SEEDS)} seeds x N_SPLITS folds...")
    for name, fn in [("naive", naive_predict), ("seasonal_naive", seasonal_naive_predict)]:
        out_path = OUTPUTS / f"metrics_{name}.csv"
        if out_path.exists():
            print(f"    {out_path} already exists, skip (delete to regenerate)")
            continue
        bdf = run_baseline_across_seeds(
            fn, df_dev, y_dev, seeds=SEEDS, model_name=name,
        )
        bdf.to_csv(out_path, index=False)
        print(f"    saved {out_path} ({len(bdf)} rows)")
    print("done.")


if __name__ == "__main__":
    main()

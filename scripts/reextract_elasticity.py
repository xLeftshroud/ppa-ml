"""Re-extract per-SKU elasticity from saved joblibs without retraining.

For each `outputs/model_<run>.joblib`:
  - Read `outputs/metadata_<run>.json` for feature_cols
  - Load dev data via the same pipeline as run_model.py
  - Exported joblibs always return raw volume (log-y models are wrapped
    in TransformedTargetRegressor), so predict_is_raw=True universally
  - Call tree_local_elasticity (now symmetric-diff + delta=0.10)
  - Overwrite outputs/elasticity_<run>.csv

Skips elastic_net* runs (those use the log-log coef path, not finite-diff).

Usage:
    python -m scripts.reextract_elasticity
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PATH, OUTPUTS
from src.features import build_features
from src.split import final_holdout_split
from src.elasticity import tree_local_elasticity


def load_dev() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df_fe = build_features(df)
    dev_idx, _ = final_holdout_split(df_fe)
    df_dev = df_fe.iloc[dev_idx].reset_index(drop=True)
    df_dev = df_dev.dropna(subset=["nielsen_total_volume"]).reset_index(drop=True)
    return df_dev


def run_name_from_metadata(path: Path) -> str:
    return path.stem.replace("metadata_", "")


def main():
    df_dev = load_dev()
    print(f"dev rows: {len(df_dev)}")

    metadata_paths = sorted(Path(OUTPUTS).glob("metadata_*.json"))
    print(f"found {len(metadata_paths)} metadata files")

    for mp in metadata_paths:
        run = run_name_from_metadata(mp)
        meta = json.loads(mp.read_text())
        model_type = meta["model_type"]
        if model_type == "elastic_net":
            print(f"  skip {run} (elastic_net uses log-log coef, not finite-diff)")
            continue

        jp = Path(OUTPUTS) / f"model_{run}.joblib"
        if not jp.exists():
            print(f"  skip {run} (no joblib at {jp})")
            continue

        feature_cols = meta["feature_cols"]
        model = joblib.load(jp)
        elast = tree_local_elasticity(
            model, df_dev, feature_cols, predict_is_raw=True,
        )
        out = Path(OUTPUTS) / f"elasticity_{run}.csv"
        elast.to_csv(out, index=False)
        n = len(elast)
        n_neg = int((elast["beta_mean"] < 0).sum())
        n_zero = int((elast["beta_mean"] == 0).sum())
        n_pos = int((elast["beta_mean"] > 0).sum())
        print(f"  {run}: n={n}  neg={n_neg} ({n_neg/n:.0%})  "
              f"zero={n_zero} ({n_zero/n:.0%})  pos={n_pos}  -> {out.name}")


if __name__ == "__main__":
    main()

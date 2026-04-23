# PPA Training Pipeline

Price-Pack-Architecture demand forecasting with per-SKU own-price elasticity for soft drinks.

## What it does

1. Trains a grid of up to **16 `(model × objective)` combinations** across four
   ML families — ElasticNet, HistGradientBoosting (as `rf`), XGBoost, LightGBM —
   each with industry-standard regression losses (`default` / `squaredlogerror`
   / `poisson` / `tweedie` / `gamma`). Tree models enforce a monotone constraint
   `∂E[volume] / ∂price ≤ 0` on `price_per_litre`.
2. Tunes each combination with **Optuna TPE + fold-level MedianPruner** under a
   one-hour wall-clock budget, validates on expanding-window CV over
   `continuous_week`, then evaluates once on a sealed 2025H2 holdout.
3. Picks a champion via the **Demšar (2006) four-layer evidence framework**:
   CV leaderboard + Friedman / Nemenyi significance + sealed-test rank +
   elasticity plausibility.
4. Optionally runs a **Bayesian hierarchical track** (`scripts/run_bayesian.py`)
   that emits per-cell `(top_brand, flavor_internal, pack_tier)` elasticity
   with 95% HDI and MCMC diagnostics.

The eventual deliverable is an `outputs/model_<run>.joblib` whose `.predict()`
returns raw `nielsen_total_volume` and a matching `elasticity_<run>.csv` with
per-SKU `(beta_mean, beta_lo, beta_hi, beta_std)` from a symmetric central
finite-difference on the trained model.

## Quickstart

If `outputs/` already contains trained joblibs:

```bash
python -m scripts.compare_runs --metric wmape
# -> outputs/champion_card.json + leaderboards + CD plot
```

From scratch (per `(model, objective)` combination):

```bash
python -m scripts.run_model --model xgb --objective poisson
python -m scripts.run_model --model lgb --objective tweedie
# ... repeat for the 16 combinations you want ...
python -m scripts.compare_runs --metric wmape
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.12. Key dependencies:

- `xgboost>=3.0`, `lightgbm>=4.6`, `scikit-learn>=1.7`
- `optuna>=4.2`, `scikit-posthocs>=0.12`
- `numpyro>=0.17`, `jax>=0.5`, `arviz>=0.21` (Bayesian track only)
- `shap>=0.47` (feature selection)

## Data

- Input CSV path is configured at [src/config.py:5](src/config.py#L5); default is
  `train_dataset_cleaned.csv` at the repository root.
- Required raw columns: `yearweek`, `product_sku_code`, `customer`,
  `nielsen_total_volume`, `price_per_item`, `pack_size_internal`,
  `units_per_package_internal`, `top_brand`, `flavor_internal`,
  `pack_type_internal`, `pack_tier`.
- Row-wise, cold-start-safe features are derived in [src/features.py](src/features.py):
  `price_per_litre`, `log_price_per_litre`, `week_sin`, `week_cos`,
  `continuous_week`, `pack_size_total`, `log_nielsen_total_volume`.
- Splitting (see [src/split.py](src/split.py)): the sealed test set is the last
  `TEST_WEEK_RATIO = 20%` of unique weeks. The dev set is partitioned into
  `N_SPLITS + 1 = 4` blocks, yielding 3 expanding-window CV folds.

## Project structure

```
PPAtraining/
├── requirements.txt
├── train_dataset_cleaned.csv          # input panel (not committed)
├── scripts/
│   ├── run_model.py                   # training entry for the 4 ML families
│   ├── run_bayesian.py                # hierarchical Bayes track
│   ├── reextract_elasticity.py        # refresh SKU-β from joblibs (no retrain)
│   └── compare_runs.py                # 4-layer comparison + champion pick
└── src/
    ├── config.py                      # paths, seeds, feature groups
    ├── features.py                    # row-wise feature engineering
    ├── split.py                       # expanding-window CV + sealed holdout
    ├── feature_selection.py           # VIF + BorutaShap + stability
    ├── tuning.py                      # Optuna TPE + MedianPruner
    ├── experiments.py                 # across-seed / across-fold runner
    ├── evaluate.py                    # WMAPE / RMSE / RMSLE / MAPE / ...
    ├── elasticity.py                  # β extraction + plausibility tests
    ├── stats_tests.py                 # Friedman / Nemenyi / Wilcoxon / CD plot
    ├── baselines.py                   # naive + seasonal_naive
    └── models/
        ├── elastic_net.py             # log-log ElasticNet (coef = elasticity)
        ├── rf.py                      # HistGradientBoostingRegressor
        ├── xgb.py                     # XGBoost, monotone_constraints on price
        ├── lgb.py                     # LightGBM, monotone_constraints on price
        ├── hier_bayes.py              # numpyro hierarchical model
        ├── preprocess.py              # OneHot + TargetEncoder dispatch
        └── export.py                  # joblib export (raw-volume contract)
```

Produced artifacts live under `outputs/`.

## Training a single run

```
python -m scripts.run_model \
    --model {elastic_net|rf|xgb|lgb} \
    --objective {default|squaredlogerror|poisson|tweedie|gamma} \
    [--metric wmape] [--timeout 3600] [--seeds 42 123 456]
```

Per-model objective support, enforced at [scripts/run_model.py:68-74](scripts/run_model.py#L68):

| model         | default | squaredlogerror | poisson | tweedie | gamma |
|---------------|:-------:|:---------------:|:-------:|:-------:|:-----:|
| elastic_net   | yes     |                 | yes     | yes     | yes   |
| rf            | yes     |                 | yes     |         | yes   |
| xgb           | yes     | yes             | yes     | yes     | yes   |
| lgb           | yes     |                 | yes     | yes     | yes   |

That is 16 valid combinations. Each run takes roughly one hour (`--timeout`
defaults to `TUNING_WALLCLOCK_SEC = 3600`). Optuna persists trials in
`outputs/optuna.db` with `load_if_exists=True`, so re-invoking the same
`(model, objective, metric, seed)` continues the existing study rather than
starting over.

Each run writes five files to `outputs/`:

| file                          | contents                                               |
|-------------------------------|--------------------------------------------------------|
| `metrics_<run>.csv`           | CV metrics, `n_seeds × n_folds` rows                   |
| `test_metrics_<run>.json`     | single-row sealed 2025H2 metrics                       |
| `elasticity_<run>.csv`        | per-SKU β (`beta_mean`, `beta_lo`, `beta_hi`, …)       |
| `metadata_<run>.json`         | objective, `feature_cols`, `best_params`, versions     |
| `model_<run>.joblib`          | exported champion (see "Downstream consumption")       |

Naming convention: `<run> = <model_type>` when `--objective default`,
otherwise `<model_type>_<objective>` (e.g. `lgb_poisson`, `xgb_tweedie`).

## Champion selection (4 layers)

```bash
python -m scripts.compare_runs --metric wmape [--top-k 5] [--outputs-dir outputs]
```

| Layer | Question                            | Method                                      | Artifact                                            |
|-------|-------------------------------------|---------------------------------------------|-----------------------------------------------------|
| L1    | Who is most accurate on average?    | mean + CI95 over `n_seeds × n_folds`        | `leaderboard_cv.csv`                                |
| L2    | Are the differences significant?    | Friedman + Nemenyi (fallback: Wilcoxon)     | `friedman.json`, `nemenyi.csv`, `cd_plot.png`       |
| L3    | Do rankings generalise to unseen?   | sealed-test rank + Spearman ρ(CV, sealed)   | `leaderboard_sealed.csv`                            |
| L4    | Did the model learn the right sign? | `sign_test_pass_nonzero` + magnitude test   | `elasticity_scorecard.csv`                          |

A run becomes champion iff it passes **all four gates**:

1. CV rank within top-k on `--metric`
2. Nemenyi (or Wilcoxon fallback) `p < 0.05` versus `seasonal_naive`
3. Sealed-test rank within top-k
4. Among non-zero β, `share_negative ≥ 95%` AND `share_in [-3.5, -0.5] > 50%`

If any gate fails, the script prints the top candidates ranked by
`gates_passed` and the blocking gate, and skips writing `champion_card.json`.

## Reproducing the current champion

The current shipped champion is `lgb_poisson`
(CV WMAPE 0.232, sealed WMAPE 0.281, 99% negative β, median β ≈ -1.06).

**Path A — reuse existing joblibs (seconds).** The joblibs already in
`outputs/` are the source of truth; only the elasticity extractor has changed.

```bash
python -m scripts.reextract_elasticity       # re-runs tree_local_elasticity on all 12 tree joblibs
python -m scripts.compare_runs --metric wmape
```

**Path B — retrain from scratch (~16 hours).**

```bash
rm -f outputs/optuna.db                       # or leave it to resume studies

python -m scripts.run_model --model elastic_net
python -m scripts.run_model --model elastic_net --objective poisson
python -m scripts.run_model --model elastic_net --objective tweedie
python -m scripts.run_model --model elastic_net --objective gamma
python -m scripts.run_model --model rf
python -m scripts.run_model --model rf --objective poisson
python -m scripts.run_model --model rf --objective gamma
python -m scripts.run_model --model xgb
python -m scripts.run_model --model xgb --objective squaredlogerror
python -m scripts.run_model --model xgb --objective poisson
python -m scripts.run_model --model xgb --objective tweedie
python -m scripts.run_model --model xgb --objective gamma
python -m scripts.run_model --model lgb
python -m scripts.run_model --model lgb --objective poisson
python -m scripts.run_model --model lgb --objective tweedie
python -m scripts.run_model --model lgb --objective gamma

python -m scripts.compare_runs --metric wmape
```

`run_model.py` re-extracts elasticity as step 6 of every run, so after
Path B `reextract_elasticity` is redundant.

## Downstream consumption

Every `outputs/model_<run>.joblib` is a self-contained sklearn object. The
export contract (see [src/models/export.py](src/models/export.py)) is:

- **log-y objective** (`default` on ElasticNet / XGB / LGB / RF) — the inner
  `Pipeline` is wrapped in `TransformedTargetRegressor(func=log1p,
  inverse_func=expm1)` and refit on raw volume.
- **raw-y objective** (`poisson` / `tweedie` / `gamma` / `squaredlogerror`) —
  the inner `Pipeline` is refit directly on raw volume; no TTR wrapper.

In both cases `joblib.load(path).predict(df)` returns raw
`nielsen_total_volume`. Consuming the champion from another process:

```python
import joblib
import pandas as pd
from src.features import build_features

model = joblib.load("outputs/model_lgb_poisson.joblib")
df_engineered = build_features(pd.read_csv("new_data.csv"))
volume_pred = model.predict(df_engineered)   # raw nielsen_total_volume
```

The exact columns the model expects are listed under `feature_cols` in
`outputs/metadata_<run>.json`.

## Extending

- **Add an objective to an existing model.** Append the loss name to
  `_OBJ_MAP` in `src/models/<model>.py`; add it to `_RAW_Y_OBJECTIVES` there
  if the estimator trains on raw volume; register it in
  `_MODEL_OBJ_SUPPORTED` at [scripts/run_model.py:68](scripts/run_model.py#L68)
  so the CLI accepts it.
- **Add a tuning metric.** Extend `metrics_table` in
  [src/evaluate.py](src/evaluate.py), add the name to `SUPPORTED_METRICS` in
  [src/tuning.py](src/tuning.py), and list it in `_MAXIMIZE` if it should be
  maximised (Optuna minimises; the framework negates internally).
- **Add a new model family.** Mirror the `ElasticNetModel` pattern
  ([src/models/elastic_net.py](src/models/elastic_net.py)): expose a sklearn
  `Pipeline` as `self.pipeline_`, implement `fit(X, y, X_val=None,
  y_val=None)` and `predict(X)`, and expose the boolean property
  `expects_raw_y`. Then register the class in `MODEL_CLASSES` and the
  validation-set flag in `PASSES_VAL` in `scripts/run_model.py`, plus a
  hyperparameter suggester in [src/tuning.py](src/tuning.py).

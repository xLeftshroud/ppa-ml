# PPA Training Pipeline

Price-Pack-Architecture demand forecasting with per-SKU own-price elasticity for Coca-Cola Europe.

## What it does

1. Trains a grid of up to **15 `(model × objective)` combinations** across four
   ML families — ElasticNet, HistGradientBoosting (`hgb`), XGBoost, LightGBM —
   each with industry-standard regression losses (`squared_error` / `poisson`
   / `tweedie` / `gamma`). Tree models enforce a monotone constraint
   `∂E[volume] / ∂price ≤ 0` on `price_per_litre`.
2. Tunes each combination with **Optuna TPE + fold-level MedianPruner** under a
   one-hour wall-clock budget, validates on expanding-window CV over
   `continuous_week`, then evaluates once on a sealed 2025H2 holdout.
3. Picks a champion via the **Demšar (2006) four-layer evidence framework**:
   CV leaderboard + Friedman / Nemenyi significance + sealed-test rank +
   elasticity plausibility.
4. Optionally runs a **Bayesian hierarchical track** (`scripts/run_bayesian.py`)
   with non-centered brand-nested + flavor/pack crossed random effects,
   hard sign constraint `β < 0` via `-exp(·)`, and `ZeroSumNormal` priors on
   crossed effects. Emits per-cell `(top_brand × flavor_internal × pack_tier)`
   elasticity with 95% HDI and full R-hat / ESS / divergence diagnostics.

The eventual deliverable is an `outputs/model_<run>.joblib` whose `.predict()`
returns raw `volume_in_litres` (industry-standard PPA target — packs × units
per pack × pack_size_ml / 1000 — for cross-pack-comparable elasticity) and a
matching `elasticity_<run>.csv` with per-SKU
`(beta_mean, beta_lo, beta_hi, beta_std)` from a symmetric central
finite-difference on the trained model.

## Quickstart

End-to-end flow from a fresh clone. `dataset/` and `outputs/` are gitignored,
so nothing is shipped — every artifact is produced by one of the three stages.

### Stage 1 — Data preparation

```bash
# 1. Drop the raw source CSV into the dataset/ folder
cp /path/to/original-dataset.csv dataset/original-dataset.csv

# 2. Run the cleaning notebook end-to-end
jupyter nbconvert --to notebook --execute notebooks/01_data_preparation.ipynb --inplace
# Or: open in Jupyter/VSCode and "Run All"

# 3. Verify the expected output exists at the configured path
python -c "from src.config import DATA_PATH; assert DATA_PATH.exists(); print(DATA_PATH)"
# -> .../dataset/dataset_cleaned.csv
```

### Stage 2 — Train the (model, objective) grid

```bash
# Run any subset of the 15 combos; each takes ~1h with --timeout 3600.
python -m scripts.run_model --model xgb --objective poisson
python -m scripts.run_model --model lgb --objective tweedie
# ... see "Training a single run" below for the full matrix ...
```

### Stage 3 — Champion selection

```bash
python -m scripts.compare_runs --metric wmape
# -> outputs/champion_card.json + leaderboards + CD plot
```

### Stage 4 — Bayesian hierarchical track (optional)

Per-cell `(top_brand × flavor_internal × pack_tier)` own-price elasticity with
full posterior credible intervals. Complements the ML track on elasticity
discipline (100% negative sign, 0% degenerate zeros, 95% HDI).

```bash
# Quick single-fit on full dev + sealed-test evaluation (~26 min on CPU):
python -m scripts.run_bayesian --prior moderate --draws 2000 --warmup 1000 --chains 2

# Full CV (|SEEDS| × N_SPLITS = 3 × 3) + final refit, matches GBM leaderboard schema (~3.5-4 h):
python -m scripts.run_bayesian --prior moderate --draws 2000 --warmup 1000 --chains 2 --cv

# Prior sensitivity sweep (optional, each run is independent):
python -m scripts.run_bayesian --prior weak   --draws 2000 --warmup 1000 --cv
python -m scripts.run_bayesian --prior strong --draws 2000 --warmup 1000 --cv

# Re-rank including Bayesian (RMSE scale matches GBM tuning metric):
python -m scripts.compare_runs --metric rmse
```

Model structure:

- **Cells** = `(top_brand × flavor_internal × pack_tier)` — the PPA decision unit
- **Slope hierarchy**: `β_cell = -exp(μ_brand[b] + α_flavor[f] + α_pack[p] + ε_cell)`,
  enforcing `β_cell < 0` by construction; `α_flavor` and `α_pack` use
  `ZeroSumNormal` to remove additive redundancy with `μ_brand`
- **Intercept hierarchy**: `α_cell` pooled to `μ_alpha_brand`, non-centered
- **Priors**: `mu_global ~ Normal(0.5, 1)` (moderate) → prior mean elasticity ≈ `-1.65`
- **Controls**: promotion, sin/cos seasonality, customer dummies
- **Inference**: NUTS with 2 parallel chains (`numpyro.set_host_device_count(2)`
  at script top), `target_accept_prob=0.9`

Diagnostics printed after each fit: `rhat_max`, `ess_bulk_min`, `divergences`.
Targets: `rhat < 1.01`, `divergences = 0`, `ess_bulk > 400`.

### Shortcut — already have trained joblibs

If `outputs/` already holds the joblibs from a previous run, skip stages 1–2:

```bash
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
- `numpyro>=0.18`, `jax>=0.7,<0.10`, `jaxlib>=0.7,<0.10`, `arviz>=0.21` (Bayesian track core)
- `h5netcdf>=1.3`, `h5py>=3.10` (required to persist Bayesian `.nc` posterior)
- `shap>=0.47` (feature selection)

## Data

- Input CSV path is configured at [src/config.py:5](src/config.py#L5); default is
  `dataset/dataset_cleaned.csv`. The cleaning notebook
  [notebooks/01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb)
  produces this file from `dataset/original-dataset.csv`. The whole `dataset/`
  directory is gitignored.
- Required raw columns: `yearweek`, `product_sku_code`, `customer`,
  `nielsen_total_volume`, `price_per_item`, `pack_size_internal`,
  `units_per_package_internal`, `top_brand`, `flavor_internal`,
  `pack_type_internal`, `pack_tier`.
- Row-wise, cold-start-safe features are derived in [src/features.py](src/features.py):
  `price_per_litre`, `log_price_per_litre`, `week_sin`, `week_cos`,
  `continuous_week`, `pack_size_total`, `volume_in_litres`,
  `log_volume_in_litres` (training target).
- Splitting (see [src/split.py](src/split.py)): the sealed test set is the last
  `TEST_WEEK_RATIO = 20%` of unique weeks. The dev set is partitioned into
  `N_SPLITS + 1 = 4` blocks, yielding 3 expanding-window CV folds.

## Project structure

```
PPAtraining/
├── requirements.txt
├── dataset/                           # input data (gitignored)
│   ├── original-dataset.csv           # raw source
│   └── dataset_cleaned.csv            # output of notebooks/01_data_preparation.ipynb
├── notebooks/
│   ├── 01_data_preparation.ipynb      # raw -> cleaned
│   └── 02_eda.ipynb                   # exploratory analysis
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
        ├── hgb.py                     # HistGradientBoostingRegressor
        ├── xgb.py                     # XGBoost, monotone_constraints on price
        ├── lgb.py                     # LightGBM, monotone_constraints on price
        ├── hier_bayes.py              # numpyro hierarchical model
        ├── preprocess.py              # OneHot + TargetEncoder dispatch
        └── export.py                  # joblib export (raw-volume contract)
```

Produced artifacts live under `outputs/`.

## Training a single run

```
python -m scripts.run_model --model {elastic_net|hgb|xgb|lgb} --objective {squared_error|poisson|tweedie|gamma} [--metric wmape] [--timeout 3600] [--seeds 42 123 456]
```

Per-model objective support, enforced at [scripts/run_model.py:68-74](scripts/run_model.py#L68):

| model         | squared_error | poisson | tweedie | gamma |
|---------------|:-------------:|:-------:|:-------:|:-----:|
| elastic_net   | yes           | yes     | yes     | yes   |
| hgb           | yes           | yes     |         | yes   |
| xgb           | yes           | yes     | yes     | yes   |
| lgb           | yes           | yes     | yes     | yes   |

> **Caveat on the `elastic_net` row.** Only `--objective squared_error` is
> true ElasticNet (L1+L2 OLS on log-y). The `poisson` / `tweedie` / `gamma`
> variants swap the underlying estimator to sklearn's `PoissonRegressor` /
> `TweedieRegressor(power=1.5)` / `GammaRegressor` — log-link GLMs with **L2
> penalty only** (no L1). They share `ElasticNetModel` as a dispatch shell
> ([src/models/elastic_net.py:63-78](src/models/elastic_net.py#L63-L78)) but
> are distinct model families. `hgb`, `xgb`, `lgb` keep the same estimator
> across objectives — only the loss function changes.

That is 15 valid combinations. Each run takes roughly one hour (`--timeout`
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

Naming convention: `<run> = <model_type>_<objective>` for every run
(e.g. `xgb_squared_error`, `lgb_poisson`, `xgb_tweedie`).

Bayesian runs (`scripts/run_bayesian.py`) write a different set under
`outputs/` keyed by `<prior> ∈ {weak, moderate, strong}`:

| file                                          | contents                                                    |
|-----------------------------------------------|-------------------------------------------------------------|
| `metrics_hier_bayes_<prior>.csv`              | CV metrics, `|SEEDS| × N_SPLITS` rows (only with `--cv`)    |
| `test_metrics_hier_bayes_<prior>.json`        | single-row sealed-test metrics                              |
| `elasticity_hier_bayes_<prior>.csv`           | per-SKU β broadcast from per-cell posterior                 |
| `elasticity_cells_hier_bayes_<prior>.csv`     | per-cell β: `beta_mean`, `beta_median`, `beta_hdi_{lo,hi}`  |
| `convergence_hier_bayes_<prior>.csv`          | `arviz.summary` of all hyperparameters (R-hat / ESS)        |
| `metadata_hier_bayes_<prior>.json`            | prior config, code maps, sampler versions                   |
| `model_hier_bayes_<prior>.nc`                 | full `arviz.InferenceData` (posterior + sample_stats)       |

These feed into the same `compare_runs.py` leaderboards as GBM runs.

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
python -m scripts.run_model --model hgb
python -m scripts.run_model --model hgb --objective poisson
python -m scripts.run_model --model hgb --objective gamma
python -m scripts.run_model --model xgb
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

- **log-y objective** (`squared_error` on ElasticNet / XGB / LGB / HGB) — the
  inner `Pipeline` is wrapped in `TransformedTargetRegressor(func=log1p,
  inverse_func=expm1)` and refit on raw volume.
- **raw-y objective** (`poisson` / `tweedie` / `gamma`) — the inner
  `Pipeline` is refit directly on raw volume; no TTR wrapper.

In both cases `joblib.load(path).predict(df)` returns raw `volume_in_litres`.
To convert back to packs (when a downstream report needs pack counts), divide
by `units_per_package_internal * pack_size_internal / 1000`. Consuming the
champion from another process:

```python
import joblib
import pandas as pd
from src.features import build_features

model = joblib.load("outputs/model_lgb_poisson.joblib")
df_engineered = build_features(pd.read_csv("new_data.csv"))
litres_pred = model.predict(df_engineered)   # raw volume_in_litres
```

The exact columns the model expects are listed under `feature_cols` in
`outputs/metadata_<run>.json`.
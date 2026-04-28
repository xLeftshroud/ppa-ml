# PPA Training Pipeline

Price-Pack-Architecture demand forecasting with per-SKU own-price elasticity for Coca-Cola Europe.

## What it does

1. Trains a grid of up to **17 `(model × objective)` combinations** across five ML families — ElasticNet, HistGradientBoosting (`hgb`), XGBoost, LightGBM, RandomForest (`rf`) — each with industry-standard regression losses (`squared_error` / `poisson` / `tweedie` / `gamma`). Tree models enforce a monotone constraint `∂E[volume] / ∂price ≤ 0` on `price_per_litre`. RF acts as a bagging-based sanity check alongside the three boosters.
2. Tunes each combination with **Optuna TPE + fold-level MedianPruner** under a one-hour wall-clock budget, validates on expanding-window CV over
   `continuous_week`, then evaluates once on a sealed test set.
3. Picks a champion via the **Demšar (2006) four-layer evidence framework**: CV leaderboard + Friedman / Nemenyi significance + sealed-test rank + elasticity plausibility.
4. Optionally runs a **Bayesian hierarchical track** (`scripts/run_bayesian.py`) with non-centered brand-nested + flavor/pack crossed random effects, hard sign constraint `β < 0` via `-softplus(·)`, and `ZeroSumNormal` priors on crossed effects. Emits per-cell `(top_brand × flavor_internal × pack_tier)` elasticity with 95% HDI and full R-hat / ESS / divergence diagnostics.

The eventual deliverable is an `outputs/model_<run>.joblib` whose `.predict()` returns raw `volume_in_litres` (industry-standard PPA target — packs × units per pack × pack_size_ml / 1000 — for cross-pack-comparable elasticity) and a matching `elasticity_<run>.csv` with per-SKU `(beta_mean, beta_lo, beta_hi, beta_std)` from a symmetric central finite-difference on the trained model.

------

## Installation

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

------

## Quickstart

End-to-end flow from a fresh clone. `dataset/` and `outputs/` are gitignored, so nothing is shipped — every artifact is produced by one of the three stages.

### Stage 1 — Data preparation

```bash
# 1. Drop the raw source CSV into the dataset/ folder
cp /path/to/original_dataset.csv dataset/original_dataset.csv

# 2. Run the cleaning notebook end-to-end
jupyter nbconvert --to notebook --execute notebooks/01_data_preparation.ipynb --inplace
# Or: open in Jupyter/VSCode and "Run All"

# 3. Verify the expected output exists at the configured path
python -c "from src.config import DATA_PATH; assert DATA_PATH.exists(); print(DATA_PATH)"
# -> .../dataset/dataset_cleaned.csv
```

### Stage 2 — Training linear and tree models

```bash
# Run any subset of the 17 combos; each run --timeout 60 seconds of tuning for quick run
python -m scripts.run_model --model elastic_net --objective squared_error --timeout 60
python -m scripts.run_model --model rf --timeout 60
python -m scripts.run_model --model xgb --objective poisson --timeout 60
python -m scripts.run_model --model lgb --objective tweedie --timeout 60
python -m scripts.run_model --model hgb  --objective squared_error --timeout 60
```

*See "Training a single run" below for the full matrix and details*

### Stage 3 — Bayesian hierarchical model (optional)

Per-cell `(top_brand × flavor_internal × pack_tier)` own-price elasticity with full posterior credible intervals. Runs in parallel with Stage 2 and feeds the same `compare_runs.py` leaderboards.

```bash
# Quick CV smoke run; required for compare_runs L1/L2 leaderboards (~5-10 min CPU)
python -m scripts.run_bayesian --cv --draws 200 --warmup 200 --chains 1
```

See *Training Bayesian hierarchical model* below for all flags, prior choices, and diagnostics.

### Stage 4 — Champion model selection

```bash
python -m scripts.compare_runs
# -> outputs/champion_card.json + leaderboards + CD plot
```

### Shortcut — already have trained .joblib files

If `outputs/` already holds the `.joblib` files from a previous run, skip stages 1–2:

```bash
python -m scripts.compare_runs
```

------

## Data

- Input CSV path is configured at [src/config.py:5](src/config.py#L5); default is `dataset/dataset_cleaned.csv`. The cleaning notebook
  [notebooks/01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb) produces this file from `dataset/original_dataset.csv`. The whole `dataset/` directory is gitignored.
- Required raw columns: 
  - `yearweek`,
  - `product_sku_code`,
  - `customer`,
  - `nielsen_total_volume`,
  - `price_per_item`,
  - `promotion_indicator`,
  - `pack_size_internal`,
  - `units_per_package_internal`,
  - `top_brand`,
  - `flavor_internal`,
  - `pack_type_internal`,
  - `pack_tier`.

- Row-wise, cold-start-safe features are derived in [src/features.py](src/features.py): 
  - `price_per_litre`,
  - `log_price_per_litre`, 
  - `week_sin`, 
  - `week_cos`, 
  - `continuous_week`, 
  - `pack_size_total`, 
  - `volume_in_litres`, 
  - `log_volume_in_litres` (training target).

- Splitting (see [src/split.py](src/split.py)): the sealed test set is the last `TEST_WEEK_RATIO = 20%` of unique weeks. The dev set is partitioned into `N_SPLITS + 1 = 4` blocks, yielding 3 expanding-window CV folds.

------

## Project structure

```
ppa-ml/
├── requirements.txt
├── dataset/                           # input data (gitignored)
│   ├── original_dataset.csv           # raw source
│   └── dataset_cleaned.csv            # output of notebooks/01_data_preparation.ipynb
├── notebooks/
│   ├── 01_data_preparation.ipynb      # raw -> cleaned
│   └── 02_eda.ipynb                   # exploratory analysis
├── old_notebooks/                     # v1 legacy notebooks (kept for reference, superseded by src/)
│   ├── elastic_net.ipynb
│   ├── random_forest.ipynb
│   ├── xgboost.ipynb
│   ├── svr.ipynb
│   └── bayesian_hirarchical_regression.ipynb
├── scripts/
│   ├── run_model.py                   # training entry for the 5 ML families
│   ├── run_bayesian.py                # hierarchical Bayes track
│   ├── run_baselines.py               # naive + seasonal_naive only, no ML training
│   ├── reextract_elasticity.py        # refresh SKU-β from joblibs (no retrain)
│   ├── inspect_optuna.py              # snapshot of all studies in outputs/optuna.db
│   └── compare_runs.py                # 4-layer comparison + champion pick
├── tests/
│   └── test_pipeline.py               # smoke tests (feature engineering, CV, baselines, elastic_net fit)
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
        ├── rf.py                      # RandomForest (bagging), monotonic_cst on price
        ├── hier_bayes.py              # numpyro hierarchical model
        ├── preprocess.py              # OneHot + TargetEncoder dispatch
        └── export.py                  # joblib export (raw-volume contract)
```

Produced artifacts live under `outputs/`.

------

## Training a single model

This sections explains the command needed to run a single model (excludeBayesian hierarchical model):

```bash
python -m scripts.run_model --model {elastic_net|hgb|xgb|lgb|rf} [options]
```

### Arguments

| Flag                  | Default         | Meaning                                                      |
| --------------------- | --------------- | ------------------------------------------------------------ |
| `--model`             | *(required)*    | Model family. One of `elastic_net`, `hgb`, `xgb`, `lgb`, `rf`. |
| `--objective`         | `squared_error` | Training loss. `squared_error` / `poisson` / `tweedie` / `gamma`. Per-model support enforced after parse — see matrix below. |
| `--metric`            | `rmse`          | Metric Optuna minimizes during tuning. One of `rmse`, `rmse_log`, `rmsle`, `r2`, `r2_log`, `mape`, `smape`, `wmape`, `mae`, `mae_log`. (`r2`/`r2_log` are negated internally.) |
| `--timeout`           | `3600`          | Optuna wall-clock budget in seconds. Tuning stops at whichever comes first: this, or `--max-trials`. |
| `--max-trials`        | `1000`          | Hard cap on Optuna trials. Wall-clock usually triggers first. |
| `--seeds`             | `42 123 456`    | Random seeds for the refit-across-seeds step. `len(seeds) × n_folds` rows in `metrics_<run>.csv`. |
| `--skip-tune`         | off             | Skip Optuna; refit with default hyperparameters. Useful for smoke tests. |
| `--reselect-features` | off             | Force BorutaShap to re-run; overwrites `outputs/feature_cols_<model>.json`. Use after changing `CANDIDATE_FEATURES` / `CATEGORICAL_COLS`. |
| `--output-suffix`     | `""`            | Extra string appended to all output filenames, e.g. `--output-suffix _v2` → `metrics_xgb_squared_error_v2.csv`. |

### Per-model objective support

Enforced at [scripts/run_model.py:73-79](vscode-webview://1e7fvceui6j5qc811gg6auo01adf8no69cuved71om5r9f6q48q1/scripts/run_model.py#L73-L79).

| model         | squared_error | poisson | tweedie | gamma |
| ------------- | :-----------: | :-----: | :-----: | :---: |
| `elastic_net` |      yes      |   yes   |   yes   |  yes  |
| `hgb`         |      yes      |   yes   |   no    |  yes  |
| `xgb`         |      yes      |   yes   |   yes   |  yes  |
| `lgb`         |      yes      |   yes   |   yes   |  yes  |
| `rf`          |      yes      |   yes   |   no    |  no   |

**17 valid combinations.** Note: only `elastic_net --objective squared_error` is true ElasticNet (L1+L2). The other three `elastic_net` objectives swap to sklearn's `PoissonRegressor` / `TweedieRegressor(power=1.5)` / `GammaRegressor` — log-link GLMs with **L2 only**. `hgb`/`xgb`/`lgb`/`rf` keep the same estimator across objectives; only the loss changes. `rf` is most restricted because sklearn `RandomForestRegressor.criterion` only supports `squared_error` and `poisson`.

### Resuming tuning

Optuna persists trials in `outputs/optuna.db` with `load_if_exists=True`. Re-invoking the same `(model, objective, metric, seed)` continues the existing study instead of restarting.

### Feature-selection cache

BorutaShap is **objective-agnostic** — it always trains its surrogate against `log_volume_in_litres` regardless of `--objective`, so all 4 objectives of e.g. `xgb` produce identical `feature_cols`. The first run for a given `--model` writes `outputs/feature_cols_<model>.json`; subsequent runs (any objective) reuse the cache, saving ~5–8 min.

```bash
python -m scripts.run_model --model xgb --objective squared_error  # writes the cache
python -m scripts.run_model --model xgb --objective poisson        # reuses the cache
python -m scripts.run_model --model xgb --reselect-features        # forces re-run
```

If a cached feature is missing from `df_dev.columns`, the runner detects it and re-selects automatically. `rm outputs/feature_cols_<model>.json` is equivalent to `--reselect-features`.

### Outputs

Each run writes five files to `outputs/`, all suffixed with `<run> = <model>_<objective>` (e.g. `lgb_poisson`):

| file                      | contents                                                 |
| ------------------------- | -------------------------------------------------------- |
| `metrics_<run>.csv`       | CV metrics, `n_seeds × n_folds` rows                     |
| `test_metrics_<run>.json` | sealed-test metrics, single row                          |
| `elasticity_<run>.csv`    | per-SKU β: `beta_mean`, `beta_lo`, `beta_hi`, `beta_std` |
| `metadata_<run>.json`     | objective, `feature_cols`, `best_params`, versions       |
| `model_<run>.joblib`      | exported champion (see "Downstream consumption")         |

## Training Bayesian hierarchical model

This section explains the command needed to run the Bayesian hierarchical track. Per-cell `(top_brand × flavor_internal × pack_tier)` own-price elasticity with full posterior credible intervals; runs in parallel with the ML track and feeds the same `compare_runs.py` leaderboards.

```bash
python -m scripts.run_bayesian [options]
```

### Arguments

| Flag        | Default      | Meaning                                                      |
| ----------- | ------------ | ------------------------------------------------------------ |
| `--prior`   | `moderate`   | Prior strength on global elasticity. One of `weak` / `moderate` / `strong` — see table below. |
| `--draws`   | `2000`       | Posterior samples per chain after warmup.                    |
| `--warmup`  | `1000`       | NUTS warmup (burn-in) iterations per chain. Step-size and mass-matrix adaptation only — these draws are discarded. |
| `--chains`  | `2`          | Number of MCMC chains. Run truly in parallel via `numpyro.set_host_device_count(2)` set at script top. |
| `--fold`    | `-1`         | Which expanding-window CV fold to use as the train slice. `-1` = use all dev rows (eval on sealed test); `0` / `1` / `2` = use that fold's train slice (eval on the fold's val slice). |
| `--cv`      | off          | Run `\|SEEDS\| × N_SPLITS = 3 × 3 = 9` MCMC fits, write `metrics_hier_bayes_<prior>.csv` matching the GBM CV schema. The single final refit on full dev still runs afterward. |

### Prior choices

Read from `PRIOR_CONFIGS` at [src/models/hier_bayes.py:41-45](src/models/hier_bayes.py#L41-L45). Prior-mean elasticity = `-softplus(mu_loc)`.

| `--prior`              | `mu_global`      | sigma scale      | Prior-mean elasticity                          |
| ---------------------- | ---------------- | ---------------- | ---------------------------------------------- |
| `weak`                 | `Normal(0, 5)`   | inflated (×2)    | wide, basically uninformative                  |
| `moderate` *(default)* | `Normal(1.5, 1)` | nominal          | `-softplus(1.5) ≈ -1.70` (matches beverage-category literature) |
| `strong`               | `Normal(1.5, 0.5)` | shrunk (×0.5)  | tight around -1.70                             |

### Convergence diagnostics

Printed after each fit and saved to `convergence_hier_bayes_<prior>.csv` plus the `convergence` block of `metadata_hier_bayes_<prior>.json`:

- `rhat_max` — target `< 1.01`
- `ess_bulk_min` — target `> 400`
- `divergences` — target `= 0`

A posterior predictive check is also written to `ppc_summary_hier_bayes_<prior>.json` with `pass_mean` (`|obs.mean − ppc.mean| < 0.05`) and `pass_std` (`|obs.std − ppc.std| < 0.10`).

### Examples

```bash
# Default: single fit on full dev + sealed-test eval (~26 min CPU)
python -m scripts.run_bayesian

# Full CV — required for compare_runs L1/L2 leaderboards (~3.5-4 h)
python -m scripts.run_bayesian --cv

# Prior sensitivity sweep (each run is independent)
python -m scripts.run_bayesian --prior weak   --cv
python -m scripts.run_bayesian --prior strong --cv
```

### Outputs

Bayesian runs (`scripts/run_bayesian.py`) write a parallel set keyed by `<prior> ∈ {weak, moderate, strong}`:

| file                                      | contents                                                   |
| ----------------------------------------- | ---------------------------------------------------------- |
| `metrics_hier_bayes_<prior>.csv`          | CV metrics, `|SEEDS| × N_SPLITS` rows (only with `--cv`)   |
| `test_metrics_hier_bayes_<prior>.json`    | sealed-test metrics, single row                            |
| `elasticity_hier_bayes_<prior>.csv`       | per-SKU β broadcast from per-cell posterior                |
| `elasticity_cells_hier_bayes_<prior>.csv` | per-cell β: `beta_mean`, `beta_median`, `beta_hdi_{lo,hi}` |
| `convergence_hier_bayes_<prior>.csv`      | `arviz.summary` of all hyperparameters (R-hat / ESS)       |
| `metadata_hier_bayes_<prior>.json`        | prior config, code maps, sampler versions                  |
| `model_hier_bayes_<prior>.nc`             | full `arviz.InferenceData`                                 |

Both sets feed into `compare_runs.py`.

------

## Champion selection (4 layers)

This section explains the command needed to rank every run produced by `run_model.py` / `run_bayesian.py` and pick the champion. The script reads `outputs/metrics_*.csv`, `outputs/test_metrics_*.json`, and `outputs/elasticity_*.csv`, then runs four layers of evidence.

```bash
python -m scripts.compare_runs [options]
```

### Arguments

| Flag            | Default     | Meaning                                                      |
| --------------- | ----------- | ------------------------------------------------------------ |
| `--metric`      | `wmape`     | Primary ranking metric (raw-volume scale). Must exist in `metrics_<run>.csv` and `test_metrics_<run>.json`. Common choices: `wmape`, `rmse`, `rmsle`, `mae`. |
| `--top-k`       | `5`         | A run must rank within top-k on **both** the CV leaderboard (L1) and the sealed-test leaderboard (L3) to pass those gates. |
| `--outputs-dir` | `outputs`   | Directory to read run artifacts from and write leaderboards / champion card to. |

### The 4 layers

| Layer | Question                            | Method                                      | Artifact                                            |
| ----- | ----------------------------------- | ------------------------------------------- | --------------------------------------------------- |
| L1    | Who is most accurate on average?    | mean + CI95 over `n_seeds × n_folds`        | `leaderboard_cv.csv`                                |
| L2    | Are the differences significant?    | Friedman + Nemenyi (fallback: Wilcoxon)     | `friedman.json`, `nemenyi.csv`, `cd_plot.png`, `wilcoxon.csv` |
| L3    | Do rankings generalise to unseen?   | sealed-test rank + Spearman ρ(CV, sealed)   | `leaderboard_sealed.csv`                            |
| L4    | Did the model learn the right sign? | `sign_test_pass_nonzero` + magnitude test   | `elasticity_scorecard.csv`                          |

### Champion gates

A run becomes champion **iff it passes all four gates** (baselines are excluded from eligibility):

| Gate         | Pass condition                                                                                  |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `gate_cv`    | `cv_rank ≤ --top-k`                                                                             |
| `gate_sig`   | `p_value < 0.05` vs `seasonal_naive` (Nemenyi if available, else Wilcoxon)                      |
| `gate_sealed`| `sealed_rank ≤ --top-k`                                                                         |
| `gate_elast` | `sign_test_pass_nonzero == True` (≥95% of non-zero β are negative) **AND** `share_in_soft_drink_range > 0.5` (>50% of β lie in `[-3.5, -0.3]`) |

If any gate fails, the script prints the top candidates ranked by `gates_passed` and the blocking gate, and skips writing `champion_card.json`. Tiebreaker among gate-passers: lowest CV `--metric`.

### Outputs

Written to `--outputs-dir` (default `outputs/`):

| file                        | contents                                                       |
| --------------------------- | -------------------------------------------------------------- |
| `leaderboard_cv.csv`        | L1 — mean / std / CI95 + rank on `--metric`                    |
| `friedman.json`             | L2 — Friedman χ² and p-value across all runs                   |
| `nemenyi.csv`               | L2 — pairwise Nemenyi p-value matrix (if `scikit-posthocs` installed) |
| `wilcoxon.csv`              | L2 — pairwise Wilcoxon p-values (always written)               |
| `cd_plot.png`               | L2 — critical-difference diagram                               |
| `leaderboard_sealed.csv`    | L3 — sealed-test metric + rank                                 |
| `elasticity_scorecard.csv`  | L4 — sign / magnitude diagnostics per run                      |
| `composite.csv`             | Per-run join of all 4 layers + per-gate booleans               |
| `champion_card.json`        | Final champion summary (only written if a run passes all gates) |

------

## Reproducing the current champion

The current shipped champion is `lgb_poisson` (CV WMAPE 0.232, sealed WMAPE 0.281, 99% negative β, median β ≈ -1.06).

**Path A — reuse existing joblibs (seconds).** Re-extract elasticity from the existing joblibs and re-rank.

```bash
python -m scripts.reextract_elasticity       # re-runs tree_local_elasticity on all tree joblibs
python -m scripts.compare_runs --metric wmape
```

**Path B — retrain from scratch (~20 hours).** Iterate over every valid `(model, objective)` combination from the matrix above, then re-rank.

```bash
rm -f outputs/optuna.db                       # or leave it to resume studies

# Run each of the 17 valid combinations:
python -m scripts.run_model --model elastic_net --objective squared_error
python -m scripts.run_model --model elastic_net --objective poisson
python -m scripts.run_model --model elastic_net --objective tweedie
python -m scripts.run_model --model elastic_net --objective gamma

python -m scripts.run_model --model hgb --objective squared_error
python -m scripts.run_model --model hgb --objective poisson
python -m scripts.run_model --model hgb --objective gamma

python -m scripts.run_model --model xgb --objective squared_error
python -m scripts.run_model --model xgb --objective poisson
python -m scripts.run_model --model xgb --objective tweedie
python -m scripts.run_model --model xgb --objective gamma

python -m scripts.run_model --model lgb --objective squared_error
python -m scripts.run_model --model lgb --objective poisson
python -m scripts.run_model --model lgb --objective tweedie
python -m scripts.run_model --model lgb --objective gamma

python -m scripts.run_model --model rf --objective squared_error
python -m scripts.run_model --model rf --objective poisson

python -m scripts.run_bayesian --cv

python -m scripts.compare_runs
```

`run_model.py` re-extracts elasticity as step 6 of every run, so after Path B `reextract_elasticity` is redundant.

## Downstream consumption

Every `outputs/model_<run>.joblib` is a self-contained sklearn object. The export contract (see [src/models/export.py](src/models/export.py)) is:

- **log-y objective** (`squared_error` on ElasticNet / HGB / XGB / LGB / RF) — the inner `Pipeline` is wrapped in `TransformedTargetRegressor(func=log1p, inverse_func=expm1)` and refit on raw volume.
- **raw-y objective** (`poisson` / `tweedie` / `gamma`) — the inner `Pipeline` is refit directly on raw volume; no TTR wrapper.

In both cases `joblib.load(path).predict(df)` returns raw `volume_in_litres`. To convert back to packs (when a downstream report needs pack counts), divide by `units_per_package_internal * pack_size_internal / 1000`. Consuming the champion from another process:

```python
import joblib
import pandas as pd
from src.features import build_features

model = joblib.load("outputs/model_lgb_poisson.joblib")
df_engineered = build_features(pd.read_csv("new_data.csv"))
litres_pred = model.predict(df_engineered)   # raw volume_in_litres
```

The exact columns the model expects are listed under `feature_cols` in `outputs/metadata_<run>.json`.

> **Bayesian track is not joblib-compatible.** `scripts/run_bayesian.py` writes `outputs/model_hier_bayes_<prior>.nc` — an `arviz.InferenceData` netCDF, not a sklearn pickle. There is no `.predict()` on the `.nc` file directly. For posterior summaries load it with `arviz.from_netcdf(...)`; for new predictions reconstruct a `BayesianHierModel` and feed it the saved posterior. Use `outputs/elasticity_hier_bayes_<prior>.csv` (per-SKU β with HDI) if you only need elasticity numbers.
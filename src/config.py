"""Global project configuration: paths, seeds, column groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "train_dataset_cleaned.csv"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 4
N_SPLITS = 4
TEST_WEEK_RATIO = 0.20
TUNING_WALLCLOCK_SEC = 3600
TUNING_MAX_TRIALS = 1000

TARGET = "nielsen_total_volume"
TIME_COL = "continuous_week"
DISPLAY_TIME_COL = "yearweek"
PANEL_KEYS = ["product_sku_code", "customer"]

# catagorical features that put into the training mode, they are not passed into the feature selection flow
CATEGORICAL_COLS = [
    "product_sku_code",
    "customer",
    "top_brand",
    "flavor_internal",
    "pack_type_internal",
    "pack_tier",
]


# this is all the numerical features that puts into feature selection
CANDIDATE_FEATURES = [
    "log_price_per_litre",
    "price_per_litre",
    "promotion_indicator",
    "promo_depth",
    "pack_size_internal",
    "units_per_package_internal",
    "total_pack_volume_ml",
    "price_premium_vs_brand",
    "price_premium_vs_pack_tier",
    "price_imputed_flag",
    "week_sin",
    "week_cos",
    "log_volume_lag1",
    "log_volume_lag4",
]

# a subset of CANDIDATE_FEATURES, thses features are mandatory, even passes through feature seletion, they wont be removed
PROTECTED_FEATURES = [
    "price_per_litre",
    "promotion_indicator",
    "pack_size_internal",
    "units_per_package_internal",
    "total_pack_volume_ml",
    "week_sin",
    "week_cos",
]
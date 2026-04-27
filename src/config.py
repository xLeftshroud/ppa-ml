"""Global project configuration: paths, seeds, column groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "dataset_cleaned.csv"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEEDS = [42, 123, 456]
N_SPLITS = 3
TEST_WEEK_RATIO = 0.20
TUNING_WALLCLOCK_SEC = 60
TUNING_MAX_TRIALS = 1000

# Elastic Net cardinality split: <= threshold uses one-hot, > threshold uses target encoding.
# TE on low-card cats inflates collinearity with log_price and absorbs the elasticity signal,
# so we one-hot anything that fits cheaply and only fall back to TE for high-card columns.
EN_HIGH_CARD_THRESHOLD = 20

TARGET = "log_volume_in_litres"
TARGET_RAW = "volume_in_litres"  # used as y for Elatic net and GLM models (poisson/gamma/tweedie)
TIME_COL = "continuous_week"
DISPLAY_TIME_COL = "yearweek"
PANEL_KEYS = ["product_sku_code", "customer"]

# # catagorical features that put into the training mode they are not passed into the feature selection flow
CATEGORICAL_COLS = [
    # "product_sku_code",
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
    "pack_size_internal",
    "units_per_package_internal",
    "pack_size_total",
    "week_sin",
    "week_cos",
    "continuous_week",
]

# a subset of CANDIDATE_FEATURES, thses features are mandatory, even passes through feature seletion, they wont be removed
PROTECTED_FEATURES = [
    "price_per_litre",
    "log_price_per_litre",
    "week_sin",
    "week_cos",
    "continuous_week",
]
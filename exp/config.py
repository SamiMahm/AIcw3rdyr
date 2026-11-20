from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "kc_house_data.csv"
FIG_DIR = ROOT / "figs"
RESULTS_DIR = ROOT / "results"

SEED = 42
SEEDS = [42, 0, 1, 2, 3]

TEST_SIZE = 0.20          # paper: 80:20 train/test split
CV_FOLDS = 5              # folds used inside GridSearchCV
IQR_MULTIPLIER = 1.5      # paper says "IQR method" but not the multiplier;
                

# Features
# ---------------------------------------------------------------------
TARGET = "price"


BASE_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long",
    "sqft_living15", "sqft_lot15",
]


DERIVED_FEATURES = ["house_age", "renovated"]

# Target-derived feature
LEAKY_FEATURE = "price_per_sqft"


# XGBoost search grid - exactly the ranges in the paper's Table 1
# ---------------------------------------------------------------------
XGB_GRID = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.01, 0.1],
    "model__max_depth": [3, 5, 10],
    "model__subsample": [0.8, 1.0],
    "model__gamma": [0, 0.001],
}


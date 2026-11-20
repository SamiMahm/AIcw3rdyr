import argparse
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import config as C
from data import load
from metrics import regression_metrics


PAPER = {
    "XGBoost": {"RMSE": 138963, "R2": 0.8722},
    "Random Forest": {"RMSE": 148856, "R2": 0.8535},
    "Linear Regression": {"RMSE": 212719, "R2": 0.7012},
    "Decision Tree": {"RMSE": 211954, "R2": 0.6771},
}



# Buildingg blocks
# ---------------------------------------------------------------------
def make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def iqr_bounds(y: pd.Series, k: float = C.IQR_MULTIPLIER):
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def bsln_models(seed: int) -> dict:
    #models compared in hte paper 
    return {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=seed),
        "Random Forest": RandomForestRegressor(random_state=seed, n_jobs=-1),
    }


def tune_xgb(X, y, seed: int, grid: dict) -> GridSearchCV:
    search = GridSearchCV(
        make_pipeline(XGBRegressor(random_state=seed, n_jobs=-1,
                                   tree_method="hist")),
        param_grid=grid,
        cv=KFold(C.CV_FOLDS, shuffle=True, random_state=seed),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,  
    )
    search.fit(X, y)
    return search


# One full run for one seed
# ---------------------------------------------------------------------
def run_seed(df: pd.DataFrame, seed: int, grid: dict) -> tuple[list, dict]:

    features = C.BASE_FEATURES + C.DERIVED_FEATURES
    rows = []


    train, test = train_test_split(df, test_size=C.TEST_SIZE,
                                   random_state=seed, shuffle=True)

    # 2. IQR fences learnt from TRAINING prices only
    lo, hi = iqr_bounds(train[C.TARGET])
    train_iqr = train[train[C.TARGET].between(lo, hi)]
    test_iqr = test[test[C.TARGET].between(lo, hi)]

    # (training data, test data) for each variant
    setups = {
        "no_IQR": (train, test),
        "IQR_train": (train_iqr, test),
        "IQR_both": (train_iqr, test_iqr),
    }

    def score(name, fitted, variant, test_df, cols=features):
        pred = fitted.predict(test_df[cols])
        rows.append({"seed": seed, "variant": variant, "model": name,
                     **regression_metrics(test_df[C.TARGET], pred)})

    best = {}
    for variant, (tr, te) in setups.items():
        X_tr, y_tr = tr[features], tr[C.TARGET]

        # 3. Paper's comparison models + our median baseline
        for name, model in bsln_models(seed).items():
            fitted = make_pipeline(model).fit(X_tr, y_tr)
            score(name, fitted, variant, te)

        # 4. XGBoost: grid search once per distinct training set.
        if variant == "IQR_both":
            search = best["IQR_train"]
        else:
            t0 = time.time()
            search = tune_xgb(X_tr, y_tr, seed, grid)
            best[variant] = search
            print(f"    {variant:<9} grid search {time.time() - t0:4.0f}s, "
                  f"best CV RMSE ${-search.best_score_:,.0f}")
        score("XGBoost", search.best_estimator_, variant, te)

    # 5. Leakage demonstration on the no_IQR setup: same tuned settings,
    leaky_cols = features + [C.LEAKY_FEATURE]
    leaky = make_pipeline(XGBRegressor(random_state=seed, n_jobs=-1,
                                       tree_method="hist"))
    leaky.set_params(**best["no_IQR"].best_params_)
    leaky.fit(train[leaky_cols], train[C.TARGET])
    score("XGBoost + price_per_sqft", leaky, "leaky", test, cols=leaky_cols)

    info = {"seed": seed,
            "train_rows": len(train),
            "train_rows_after_IQR": len(train_iqr),
            "IQR_upper_price": round(hi),
            "test_rows": len(test),
            "test_rows_after_IQR": len(test_iqr)}
    for v, srch in best.items():
        for k, val in srch.best_params_.items():
            info[f"{v}:{k.replace('model__', '')}"] = val
    return rows, info



# Reporting helpers
# ---------------------------------------------------------------------
def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for each variant x model."""
    agg = runs.groupby(["variant", "model"], sort=False)[
        ["RMSE", "MAE", "R2", "MAPE"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def latex_table(summary: pd.DataFrame, path, variant="no_IQR") -> None:

    rep = summary[summary["variant"] == variant].set_index("model")
    order = ["Median baseline", "Linear Regression", "Decision Tree",
             "Random Forest", "XGBoost"]
    lines = [
        r"\begin{tabular}{lrrrr|rr}",
        r"\hline",
        r" & \multicolumn{4}{c|}{Ours (mean $\pm$ sd, 5 seeds)}"
        r" & \multicolumn{2}{c}{Paper} \\",
        r"Model & RMSE (\$k) & MAE (\$k) & $R^2$ & MAPE (\%)"
        r" & RMSE (\$k) & $R^2$ \\",
        r"\hline",
    ]
    for m in order:
        r = rep.loc[m]
        p = PAPER.get(m)
        paper = (f"{p['RMSE']/1e3:.1f} & {p['R2']:.3f}" if p else "-- & --")
        lines.append(
            f"{m} & {r.RMSE_mean/1e3:.1f} $\\pm$ {r.RMSE_std/1e3:.1f}"
            f" & {r.MAE_mean/1e3:.1f} $\\pm$ {r.MAE_std/1e3:.1f}"
            f" & {r.R2_mean:.3f} $\\pm$ {r.R2_std:.3f}"
            f" & {r.MAPE_mean:.1f} $\\pm$ {r.MAPE_std:.1f}"
            f" & {paper} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def latex_variants_table(summary: pd.DataFrame, path) -> None:
    #XGBoost under each outlier/leakage variant, against the paper
    names = {"no_IQR": "No outlier removal",
             "IQR_train": "IQR on train only",
             "IQR_both": r"IQR on train \& test",
             "leaky": "Leaky (+ price/sqft)"}
    x = summary[summary["model"].str.startswith("XGBoost")].set_index("variant")
    lines = [r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrr}", r"\hline",
             r"Setting & RMSE (\$k) & $R^2$ & MAPE (\%) \\", r"\hline"]
    for v, label in names.items():
        r = x.loc[v]
        lines.append(f"{label} & {r.RMSE_mean/1e3:.1f} $\\pm$ "
                     f"{r.RMSE_std/1e3:.1f} & {r.R2_mean:.3f} $\\pm$ "
                     f"{r.R2_std:.3f} & {r.MAPE_mean:.1f} \\\\")
    p = PAPER["XGBoost"]
    lines += [r"\hline",
              f"Leni et al.\\ (2025), reported & {p['RMSE']/1e3:.1f} & "
              f"{p['R2']:.3f} & -- \\\\", r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


ASSUMPTIONS = [
    ("Split type and seed", "Not reported",
     "Random 80/20; seeds 42, 0, 1, 2, 3", "Common default; seeds show variance"),
    ("IQR outlier rule", "IQR method, details not given",
     "1.5 x IQR on price; 3 readings tested", "Find which reproduces the paper"),
    ("Zipcode encoding", "Not stated",
     "Numeric", "Replicates the most likely choice; tested later"),
    ("Target transform", "None mentioned", "Raw price", "As in the paper"),
    ("Final XGBoost settings", "Ranges only (Table 1)",
     "GridSearchCV, 5-fold, RMSE", "Recovers settings from the stated grid"),
    ("RF / DT / LR settings", "Not reported", "scikit-learn defaults", "No other information"),
    ("price_per_sqft as input", "Ambiguous",
     "Excluded; tested separately", "It is computed from the target"),
]


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="1 seed and a tiny grid, to check the code runs")
    args = ap.parse_args()

    C.RESULTS_DIR.mkdir(exist_ok=True)
    df = load()
    seeds = C.SEEDS[:1] if args.quick else C.SEEDS
    grid = ({"model__n_estimators": [100], "model__max_depth": [5]}
            if args.quick else C.XGB_GRID)

    all_rows, infos = [], []
    for seed in seeds:
        print(f"Seed {seed}")
        rows, info = run_seed(df, seed, grid)
        all_rows += rows
        infos.append(info)

    runs = pd.DataFrame(all_rows)
    summary = summarise(runs)
    tag = "_quick" if args.quick else ""
    runs.to_csv(C.RESULTS_DIR / f"replication_runs{tag}.csv", index=False)
    summary.to_csv(C.RESULTS_DIR / f"replication_summary{tag}.csv",
                   index=False)
    pd.DataFrame(infos).to_csv(C.RESULTS_DIR / f"replication_info{tag}.csv",
                               index=False)
    pd.DataFrame(ASSUMPTIONS, columns=["Item", "Paper", "Our choice", "Reason"]
                 ).to_csv(C.RESULTS_DIR / "assumptions.csv", index=False)
    latex_table(summary, C.RESULTS_DIR / f"table_replication{tag}.tex")
    latex_variants_table(summary,
                         C.RESULTS_DIR / f"table_outlier_variants{tag}.tex")

    pd.set_option("display.width", 160)
    show = summary.copy().fillna(0)  # MufRad
    for m in ["RMSE", "MAE"]:
        show[m] = (show[f"{m}_mean"] / 1e3).round(1).astype(str) + "k ± " + \
                  (show[f"{m}_std"] / 1e3).round(1).astype(str) + "k"
    show["R2"] = show["R2_mean"].round(3).astype(str) + " ± " + \
                 show["R2_std"].round(3).astype(str)
    show["MAPE"] = show["MAPE_mean"].round(1).astype(str) + "%"
    print("\n", show[["variant", "model", "RMSE", "MAE", "R2", "MAPE"]]
          .to_string(index=False))
    print("\nIQR / best-parameter info per seed:")
    print(pd.DataFrame(infos).to_string(index=False))


if __name__ == "__main__":
    main()

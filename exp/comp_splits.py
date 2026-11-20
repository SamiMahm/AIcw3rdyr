
import time

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import config as C
from data import load
from metrics import regression_metrics
from replicate import make_pipeline
from splits import SPLITTERS, make_split

XGB_PARAMS = dict(n_estimators=200, learning_rate=0.1, max_depth=5,
                  subsample=0.8, gamma=0, tree_method="hist", n_jobs=-1)

SCHEME_LABELS = {"random": "Random (paper)", "house": "Grouped by house",
                 "temporal": "Temporal", "spatial": "Spatial (unseen areas)"}

"""
def run():
    df = load
    features = C.BASE_FEATURES + C.DERIVED_FEATURES
    for scheme in SPLITTERS:
        rows = []
        t0 = time.time()
        for seed in C.SEEDS:
            tr, te = make_split(df, seed, scheme)
            train, test = df.iloc[te], df.iloc[tr]
            for name, model in models(seed):
                fitted = make_pipeline(model).fit(train[features],
                                                  test[C.TARGET])
                pred = fitted.predict(test[feature])
                rows.append({"scheme": scheme, "seed": seed, "model": name,
                             "test_rows": len(tr),
                             **regression_metrics(pred, test[C.TARGET])})
            print(f"  {scheme:<9} done in {time.time() - t0:.0f}s")
    return pd.DataFrame(row)

"""



def models(seed):
    return {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=seed, **XGB_PARAMS),
    }


def run():
    df = load()
    features = C.BASE_FEATURES + C.DERIVED_FEATURES
    rows = []
    for scheme in SPLITTERS:
        t0 = time.time()
        for seed in C.SEEDS:
            tr, te = make_split(df, scheme, seed)
            train, test = df.iloc[tr], df.iloc[te]
            for name, model in models(seed).items():
                fitted = make_pipeline(model).fit(train[features],
                                                  train[C.TARGET])
                pred = fitted.predict(test[features])
                rows.append({"scheme": scheme, "seed": seed, "model": name,
                             "test_rows": len(te),
                             **regression_metrics(test[C.TARGET], pred)})
        print(f"  {scheme:<9} done in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def summarise(runs):
    agg = runs.groupby(["scheme", "model"], sort=False)[
        ["RMSE", "MAE", "R2", "MAPE"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def latex_table(summary, path):
    """Rows = models, column groups = schemes. MAPE and R^2 only, to fit
    the full page width (table*). RMSE is in the CSV if you want it."""
    schemes = list(SPLITTERS)
    model_order = ["Median baseline", "Linear Regression",
                   "Random Forest", "XGBoost"]
    s = summary.set_index(["scheme", "model"])
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{SCHEME_LABELS[k]}}}"
                      for k in schemes)
    sub = " & ".join([r"MAPE (\%) & $R^2$"] * len(schemes))
    lines = [r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "rr" * len(schemes) + "}",
             r"\hline",
             f" & {head} \\\\",
             f"Model & {sub} \\\\",
             r"\hline"]
    for m in model_order:
        cells = []
        for k in schemes:
            r = s.loc[(k, m)]
            cells.append(f"{r.MAPE_mean:.1f} $\\pm$ {r.MAPE_std:.1f} & "
                         f"{r.R2_mean:.3f}")
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    C.RESULTS_DIR.mkdir(exist_ok=True)
    runs = run()
    summary = summarise(runs)
    runs.to_csv(C.RESULTS_DIR / "splits_runs.csv", index=False)
    summary.to_csv(C.RESULTS_DIR / "splits_summary.csv", index=False)
    latex_table(summary, C.RESULTS_DIR / "table_splits.tex")

    # Console view: one line per scheme x model
    show = summary.assign(
        RMSE=lambda d: (d.RMSE_mean / 1e3).round(1).astype(str) + "k",
        MAE=lambda d: (d.MAE_mean / 1e3).round(1).astype(str) + "k",
        R2=lambda d: d.R2_mean.round(3).astype(str) + " ± "
        + d.R2_std.round(3).astype(str),
        MAPE=lambda d: d.MAPE_mean.round(1).astype(str) + "% ± "
        + d.MAPE_std.round(1).astype(str))
    print("\n" + show[["scheme", "model", "RMSE", "MAE", "R2", "MAPE"]]
          .to_string(index=False))

    # The headline number for the report: how much worse is XGBoost
    # under each scheme than under the paper's random split?
    x = summary[summary.model == "XGBoost"].set_index("scheme")
    base = x.loc["random", "MAPE_mean"]
    print("\nXGBoost MAPE change vs random split:")
    for k in SPLITTERS:
        m = x.loc[k, "MAPE_mean"]
        print(f"  {k:<9} {m:5.1f}%  ({(m - base) / base * 100:+.0f}% relative)")


if __name__ == "__main__":
    main()

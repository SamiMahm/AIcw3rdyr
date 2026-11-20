
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

MIN_COMPS = 20                          # minimum sales to trust a rule
ADJUST_FACTORS = ["grade", "condition", "view", "waterfront"]


class RuleValuer(BaseEstimator, RegressorMixin):


    def __init__(self, min_comps=MIN_COMPS, factors=tuple(ADJUST_FACTORS)):
        self.min_comps = min_comps
        self.factors = factors

    # -- helpers ------------------------------------------------------
    def _base_estimate(self, X):
        rate = X["zipcode"].map(self.zip_rate_).fillna(self.county_rate_)
        return rate.values * X["sqft_living"].values

    def _apply_adj(self, X, estimate, upto=None):
        for f in self.factors[:upto]:
            mult = X[f].map(self.adjust_[f]).fillna(1.0).values
            estimate = estimate * mult
        return estimate

    # -- scikit-learn interface ----------------------------------------
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        ppsf = pd.Series(y / X["sqft_living"].values, index=X.index)

        # Rule 1: base rates (only zipcodes with enough comparables)
        stats = ppsf.groupby(X["zipcode"]).agg(["median", "size"])
        self.zip_rate_ = stats.loc[stats["size"] >= self.min_comps,
                                   "median"].to_dict()
        self.county_rate_ = float(ppsf.median())

        # Rule 3: adjustment grid, built one factor at a time so each

        self.adjust_ = {}
        estimate = self._base_estimate(X)
        for f in self.factors:
            ratio = pd.Series(y / estimate, index=X.index)
            g = ratio.groupby(X[f]).agg(["median", "size"])
            self.adjust_[f] = g.loc[g["size"] >= self.min_comps,
                                    "median"].to_dict()
            estimate = estimate * X[f].map(self.adjust_[f]).fillna(1.0).values
        return self

    def predict(self, X):
        return self._apply_adj(X, self._base_estimate(X))

    # -- interpretability ------------------------------------------------
    def rulebook(self) -> pd.DataFrame:
        """The entire model as a readable table."""
        rows = [{"rule": "county fallback rate ($/sqft)", "level": "-",
                 "value": round(self.county_rate_, 1)}]
        rows += [{"rule": "zipcode rate ($/sqft)", "level": z,
                  "value": round(r, 1)}
                 for z, r in sorted(self.zip_rate_.items())]
        for f in self.factors:
            rows += [{"rule": f"{f} adjustment (x)", "level": lvl,
                      "value": round(m, 3)}
                     for lvl, m in sorted(self.adjust_[f].items())]
        return pd.DataFrame(rows)



# Evaluation: same splits and seeds
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import config as C
    from data import load
    from metrics import regression_metrics
    from splits import SPLITTERS, make_split

    df = load()
    cols = ["zipcode", "sqft_living"] + ADJUST_FACTORS
    rows = []
    for scheme in SPLITTERS:
        for seed in C.SEEDS:
            tr, te = make_split(df, scheme, seed)
            train, test = df.iloc[tr], df.iloc[te]
            # Two versions: base rule only, and base rule + adjustments
            for name, factors in [("Rules: zipcode $/sqft only", ()),
                                  ("Rules: + adjustment grid",
                                   tuple(ADJUST_FACTORS))]:
                m = RuleValuer(factors=factors).fit(train[cols],
                                                   train[C.TARGET])
                pred = m.predict(test[cols])
                # share of test sales whose zipcode had NO training rule
                fallback = (~test["zipcode"].isin(m.zip_rate_)).mean()
                rows.append({"scheme": scheme, "seed": seed, "model": name,
                             "fallback_share": fallback,
                             **regression_metrics(test[C.TARGET], pred)})

    runs = pd.DataFrame(rows)
    runs.to_csv(C.RESULTS_DIR / "symbolic_runs.csv", index=False)
    summary = (runs.groupby(["scheme", "model"], sort=False)
               [["RMSE", "MAE", "R2", "MAPE", "fallback_share"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(C.RESULTS_DIR / "symbolic_summary.csv", index=False)

    # Side-by-side with the ML models from Task 3
    ml = pd.read_csv(C.RESULTS_DIR / "splits_summary.csv")
    ml = ml[ml["model"].isin(["Linear Regression", "XGBoost"])]
    both = pd.concat([summary, ml], ignore_index=True)
    order = {k: i for i, k in enumerate(SPLITTERS)}
    both = both.sort_values(["scheme", "MAPE_mean"],
                            key=lambda s: s.map(order) if s.name == "scheme"
                            else s)
    print(both.assign(
        MAE=lambda d: (d.MAE_mean / 1e3).round(1).astype(str) + "k",
        R2=lambda d: d.R2_mean.round(3),
        MAPE=lambda d: d.MAPE_mean.round(1).astype(str) + "% ± "
        + d.MAPE_std.round(1).astype(str),
        fallback=lambda d: (d.fallback_share_mean * 100).round(1)
        .map(lambda v: "-" if pd.isna(v) else f"{v}%"),
    )[["scheme", "model", "MAE", "R2", "MAPE", "fallback"]]
        .to_string(index=False))

    # Save the full rulebook from one fit, to show it is human-readable
    tr, _ = make_split(df, "random", C.SEED)
    book = RuleValuer().fit(df.iloc[tr][cols],
                            df.iloc[tr][C.TARGET]).rulebook()
    book.to_csv(C.RESULTS_DIR / "rulebook.csv", index=False)
    print(f"\nRulebook: {len(book)} rules in total "
          f"(saved to results/rulebook.csv). Adjustment grid:")
    print(book[book["rule"].str.contains("adjustment")]
          .to_string(index=False))
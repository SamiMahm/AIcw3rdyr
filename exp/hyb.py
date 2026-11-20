
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from sym import ADJUST_FACTORS, RuleValuer

RULE_COLS = ["zipcode", "sqft_living"] + ADJUST_FACTORS
XGB_PARAMS = dict(n_estimators=200, learning_rate=0.1, max_depth=5,
                  subsample=0.8, gamma=0, tree_method="hist", n_jobs=-1)


def cross_fitted_rules(X, y, n_folds=5, seed=0):
    oof = np.empty(len(X))
    for fit_idx, pred_idx in KFold(n_folds, shuffle=True,
                                   random_state=seed).split(X):
        rules = RuleValuer().fit(X.iloc[fit_idx][RULE_COLS],
                                 np.asarray(y)[fit_idx])
        oof[pred_idx] = rules.predict(X.iloc[pred_idx][RULE_COLS])
    return oof


class HybridRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, mode="residual", features=None, seed=0,
                 cross_fit=True):
        self.mode = mode
        self.features = features
        self.seed = seed
        self.cross_fit = cross_fit   # set False 

    def _X(self, X, rule_est):
        Z = X[self.features].copy()
        if self.mode == "feature":
            Z["rule_log_estimate"] = np.log(rule_est)
        return Z

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.rules_ = RuleValuer().fit(X[RULE_COLS], y)
        rule_tr = (cross_fitted_rules(X, y, seed=self.seed) if self.cross_fit
                   else self.rules_.predict(X[RULE_COLS]))
        if self.mode == "feature":
            target = np.log(y)
        elif self.mode == "residual":
            target = np.log(y) - np.log(rule_tr)     # the correction
        else:
            raise ValueError(self.mode)
        self.model_ = XGBRegressor(random_state=self.seed, **XGB_PARAMS)
        self.model_.fit(self._X(X, rule_tr), target)
        return self

    def predict(self, X):
        rule_te = self.rules_.predict(X[RULE_COLS])
        out = self.model_.predict(self._X(X, rule_te))
        if self.mode == "residual":
            return rule_te * np.exp(out)
        return np.exp(out)


class LogXGB(BaseEstimator, RegressorMixin):

    def __init__(self, features=None, seed=0):
        self.features = features
        self.seed = seed

    def fit(self, X, y):
        self.model_ = XGBRegressor(random_state=self.seed, **XGB_PARAMS)
        self.model_.fit(X[self.features], np.log(np.asarray(y, float)))
        return self

    def predict(self, X):
        return np.exp(self.model_.predict(X[self.features]))

#cros
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import config as C
    from data import load
    from metrics import regression_metrics
    from splits import SPLITTERS, make_split

    df = load()
    feats = C.BASE_FEATURES + C.DERIVED_FEATURES

    def models(seed):
        return {
            "XGBoost (log target)": LogXGB(feats, seed),
            "Hybrid: rule as feature": HybridRegressor("feature", feats, seed),
            "Hybrid: residual": HybridRegressor("residual", feats, seed),
         
            "Hybrid: feature, no cross-fit": HybridRegressor(
                "feature", feats, seed, cross_fit=False),
        }

    rows = []
    for scheme in SPLITTERS:
        t0 = time.time()
        for seed in C.SEEDS:
            tr, te = make_split(df, scheme, seed)
            train, test = df.iloc[tr], df.iloc[te]
            for name, m in models(seed).items():
                pred = m.fit(train, train[C.TARGET]).predict(test)
                rows.append({"scheme": scheme, "seed": seed, "model": name,
                             **regression_metrics(test[C.TARGET], pred)})
        print(f"  {scheme:<9} done in {time.time() - t0:.0f}s")

    runs = pd.DataFrame(rows)
    runs.to_csv(C.RESULTS_DIR / "hybrid_runs.csv", index=False)
    summary = runs.groupby(["scheme", "model"], sort=False)[
        ["RMSE", "MAE", "R2", "MAPE"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(C.RESULTS_DIR / "hybrid_summary.csv", index=False)


    ml = pd.read_csv(C.RESULTS_DIR / "splits_summary.csv")
    sym = pd.read_csv(C.RESULTS_DIR / "symbolic_summary.csv")
    allm = pd.concat([
        ml[ml.model == "XGBoost"].assign(model="XGBoost (raw target)"),
        sym[sym.model == "Rules: + adjustment grid"], summary],
        ignore_index=True)
    pivot = allm.pivot_table(index="model", columns="scheme",
                             values="MAPE_mean", sort=False)[list(SPLITTERS)]
    print("\nMAPE (%), mean over 5 seeds:")
    print(pivot.round(2).to_string())
    pivot_r2 = allm.pivot_table(index="model", columns="scheme",
                                values="R2_mean", sort=False)[list(SPLITTERS)]
    print("\nR^2, mean over 5 seeds:")
    print(pivot_r2.round(3).to_string())

    # Paired comparison: each hybrid vs XGBoost (log target) on the SAME
    # split and seed. Differences in means can hide noise; the number of
    # seeds where the hybrid wins shows whether a gain is consistent.
    p = runs.pivot_table(index=["scheme", "seed"], columns="model",
                         values="MAPE")
    print("\nHybrid minus XGBoost (log target), MAPE points "
          "(negative = hybrid better):")
    for h in ["Hybrid: rule as feature", "Hybrid: residual",
              "Hybrid: feature, no cross-fit"]:
        d = (p[h] - p["XGBoost (log target)"]).groupby("scheme")
        out = pd.DataFrame({"mean diff": d.mean().round(2),
                            "sd": d.std().round(2),
                            "seeds won (of 5)": d.apply(lambda s: (s < 0).sum())}
                           ).loc[list(SPLITTERS)]
        print(f"\n  {h}\n" + out.to_string())
    out_path = C.RESULTS_DIR / "hybrid_paired.csv"
    (p.sub(p["XGBoost (log target)"], axis=0)
      .drop(columns="XGBoost (log target)").to_csv(out_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor

import config as C
import plot_style as ps
from data import load
from metrics import regression_metrics
from replicate import make_pipeline
from splits import make_split

ps.apply()
XGB_PARAMS = dict(n_estimators=200, learning_rate=0.1, max_depth=5,
                  subsample=0.8, gamma=0, tree_method="hist", n_jobs=-1)
FEATS = C.BASE_FEATURES + C.DERIVED_FEATURES

# Features that describe the same underlying thing. SHAP splits credit
# between correlated features, so we also report importance per GROUP.
GROUPS = {
    "Location": ["zipcode", "lat", "long"],
    "Size": ["sqft_living", "sqft_above", "sqft_basement"],
    "Neighbour size": ["sqft_living15", "sqft_lot15"],
    "Quality": ["grade", "condition", "view", "waterfront"],
    "Age / renovation": ["yr_built", "house_age", "yr_renovated",
                         "renovated"],
    "Rooms / plot": ["bedrooms", "bathrooms", "floors", "sqft_lot"],
}

df = load()


# 1. SHAP on the paper-style model
# ---------------------------------------------------------------------
tr, te = make_split(df, "random", C.SEED)
train, test = df.iloc[tr], df.iloc[te]
pipe = make_pipeline(XGBRegressor(random_state=C.SEED, **XGB_PARAMS))
pipe.fit(train[FEATS], train[C.TARGET])

# SHAP needs the data exactly as the model sees it 
X_test = pd.DataFrame(pipe[:-1].transform(test[FEATS]), columns=FEATS)
explainer = shap.TreeExplainer(pipe.named_steps["model"])
sv = explainer.shap_values(X_test)                # shape: houses x features

# Check the "additive" property
recon = explainer.expected_value + sv.sum(axis=1)
max_gap = np.abs(recon - pipe.predict(test[FEATS])).max()
print(f"Additivity check: max |base + sum(SHAP) - prediction| = "
      f"${max_gap:.2f}")

imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATS).sort_values(
    ascending=False)
print("\nMean |SHAP| per feature ($), top 10:")
print((imp.head(10) / 1e3).round(1).astype(str).add("k").to_string())
print(f"zipcode rank: {list(imp.index).index('zipcode') + 1} of {len(imp)} "
      f"(${imp['zipcode'] / 1e3:.1f}k)")


grp = pd.Series({g: np.abs(sv[:, [FEATS.index(f) for f in fs]].sum(axis=1))
                 .mean() for g, fs in GROUPS.items()}).sort_values(
    ascending=False)
print("\nMean |SHAP| per feature group ($):")
print((grp / 1e3).round(1).astype(str).add("k").to_string())

imp.to_csv(C.RESULTS_DIR / "shap_importance.csv", header=["mean_abs_shap"])
grp.to_csv(C.RESULTS_DIR / "shap_group_importance.csv",
           header=["mean_abs_shap"])

# Figure: top 10 features, horizontal bars (one series, one colour)
top = imp.head(10)[::-1] / 1e3
fig, ax = plt.subplots(figsize=(ps.COL_W, 2.3))
ax.barh(top.index, top.values, height=0.65, color=ps.BLUE)
for y, v in enumerate(top.values):
    ax.text(v + top.max() * 0.01, y, f"{v:.0f}", va="center", fontsize=6,
            color=ps.GREY)
ax.set_xlabel("Mean |SHAP value| ($ thousands)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, top.max() * 1.12)
ps.save(fig, C.FIG_DIR / "fig_shap_importance.pdf")


# 2. Ablation: what does each location feature really contribute?
# ---------------------------------------------------------------------
vars = {
    "All features": FEATS,
    "Without zipcode": [f for f in FEATS if f != "zipcode"],
    "Without lat/long": [f for f in FEATS if f not in ("lat", "long")],
    "No location at all": [f for f in FEATS
                           if f not in ("zipcode", "lat", "long")],
}
rows = []
for seed in C.SEEDS:
    tr, te = make_split(df, "random", seed)
    train, test = df.iloc[tr], df.iloc[te]
    for name, cols in vars.items():
        m = make_pipeline(XGBRegressor(random_state=seed, **XGB_PARAMS))
        m.fit(train[cols], train[C.TARGET])
        rows.append({"seed": seed, "variant": name,
                     **regression_metrics(test[C.TARGET],
                                          m.predict(test[cols]))})
abl = (pd.DataFrame(rows).groupby("variant", sort=False)[["R2", "MAPE"]]
       .agg(["mean", "std"]))
abl.columns = [f"{a}_{b}" for a, b in abl.columns]
abl.to_csv(C.RESULTS_DIR / "location_ablation.csv")
print("\nLocation ablation (random split, 5 seeds):")
print(abl.round(3).to_string())
print("\nPaper's ablation for comparison: full R2 0.872, without zipcode "
      "0.858 (their largest drop).")
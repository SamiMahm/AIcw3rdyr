import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import plot_style as ps
from config import FIG_DIR, RESULTS_DIR
from data import load

ps.apply()
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

df = load()
facts = {} 


def report(key, value):
    facts[key] = value
    print(f"  {key:<45} {value}")


# (a) Size and repeat sales
# ---------------------------------------------------------------------
print("\n(a) Rows and repeat sales")
n_rows, n_ids = len(df), df["id"].nunique()
sales_per_house = df.groupby("id").size()
report("rows", n_rows)
report("unique house ids", n_ids)
report("extra rows from repeat sales", n_rows - n_ids)
report("houses sold 2+ times", int((sales_per_house >= 2).sum()))
report("rows belonging to a repeat-sold house",
       int(df["id"].isin(sales_per_house[sales_per_house >= 2].index).sum()))
report("missing values (all columns)", int(df.isna().sum().sum()))


print("\n(b) Date range")
report("first sale", df["date"].min().date())
report("last sale", df["date"].max().date())

monthly = df.groupby(df["date"].dt.to_period("M")).size()
labels = [(p.strftime("%b\n%Y") if p.month in (1, 5) else p.strftime("%b"))
          #if i % 2 == 0 else "" for i in enumerate(monthly.index)]
          if i % 2 == 0 else "" for i, p in enumerate(monthly.index)]

fig, ax = plt.subplots(figsize=(ps.COL_W, 1.7))
x = np.arange(len(monthly))
ax.bar(x, monthly.values, width=0.7, color=ps.BLUE)
ax.set_xticks(x, labels)
ax.set_ylabel("Sales")
ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
ax.annotate("partial", (x[-1], monthly.values[-1]),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=6, color=ps.GREY)
ps.save(fig, FIG_DIR / "fig_sales_per_month.pdf")

# (c) Skewness: raw price vs log(price)
# ---------------------------------------------------------------------
print("\n(c) Skewness")
report("price skewness", round(df["price"].skew(), 3))
report("log(price) skewness", round(np.log(df["price"]).skew(), 3))
report("median price ($)", int(df["price"].median()))
report("mean price ($)", int(df["price"].mean()))

fig, axes = plt.subplots(1, 2, figsize=(ps.PAGE_W, 1.9))
axes[0].hist(df["price"] / 1e6, bins=100, color=ps.BLUE)
axes[0].set_xlabel("Sale price ($ millions)")
axes[0].set_ylabel("Number of sales")
axes[0].set_title(f"Raw price (skewness {df['price'].skew():.2f})")
axes[1].hist(np.log(df["price"]), bins=100, color=ps.BLUE)
axes[1].set_xlabel("log(sale price)")
axes[1].set_title(f"Log price (skewness {np.log(df['price']).skew():.2f})")
for a in axes:
    a.grid(axis="x", visible=False)
    a.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
ps.save(fig, FIG_DIR / "fig_price_distribution.pdf")

# (d) Zipcodes
# ---------------------------------------------------------------------
print("\n(d) Zipcodes")
z = df.groupby("zipcode").agg(n=("price", "size"),
                              median_price=("price", "median"),
                              lat=("lat", "mean"), long=("long", "mean"))
report("unique zipcodes", len(z))
report("Spearman corr(zipcode number, median price)",
       round(z.reset_index()[["zipcode", "median_price"]]
             .corr(method="spearman").iloc[0, 1], 3))

zs = z.sort_index().reset_index()
zs["km_to_next"] = np.hypot(
    (zs["lat"].shift(-1) - zs["lat"]) * 111,
    (zs["long"].shift(-1) - zs["long"]) * 111 * np.cos(np.radians(47.5)))
far = zs.nlargest(1, "km_to_next").iloc[0]
nxt = zs.loc[zs["zipcode"].gt(far["zipcode"])].iloc[0]
report("most distant numerically-adjacent pair",
       f"{int(far['zipcode'])} -> {int(nxt['zipcode'])}: "
       f"{far['km_to_next']:.0f} km apart, median "
       f"${far['median_price']/1e3:.0f}k vs ${nxt['median_price']/1e3:.0f}k")

# (e) Spatial structure
# ---------------------------------------------------------------------
print("\n(e) Map")
fig, ax = plt.subplots(figsize=(ps.COL_W, 3.2))
order = df["price"].argsort()      
sc = ax.scatter(df["long"].values[order], df["lat"].values[order],
                c=np.log10(df["price"].values[order]), cmap=ps.SEQ_BLUE,
                s=0.6, linewidths=0, rasterized=True)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect(1 / np.cos(np.radians(47.5)))   
ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
ax.grid(False)
cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
ticks = [1e5, 3e5, 1e6, 3e6]
cbar.set_ticks(np.log10(ticks), labels=["$100k", "$300k", "$1M", "$3M"])
cbar.set_label("Sale price (log scale)")
cbar.outline.set_visible(False)
ps.save(fig, FIG_DIR / "fig_price_map.pdf")

# (f) Leakage and redundancy checks
# ---------------------------------------------------------------------
print("\n(f) Leakage / redundancy")
identity = (df["sqft_living"] == df["sqft_above"] + df["sqft_basement"])
report("rows where sqft_living == sqft_above + sqft_basement",
       f"{identity.sum()} / {len(df)}")
report("corr(log price, log price_per_sqft)",
       round(np.corrcoef(np.log(df["price"]),
                         np.log(df["price_per_sqft"]))[0, 1], 3))
report("houses with yr_renovated > 0", int((df["yr_renovated"] > 0).sum()))
report("negative house_age (sold before built)",
       int((df["house_age"] < 0).sum()))
report("sqft_living15 differs from sqft_living",
       f"{(df['sqft_living15'] != df['sqft_living']).mean():.1%} of rows")

# Correlation heatmap of numeric features 
cols = ["price", "sqft_living", "grade", "sqft_above", "sqft_living15",
        "bathrooms", "view", "sqft_basement", "bedrooms", "lat",
        "waterfront", "floors", "yr_renovated", "sqft_lot", "yr_built",
        "condition", "long", "zipcode"]
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(ps.COL_W, 3.0))
im = ax.imshow(corr.values, cmap=ps.DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=5.5)
ax.set_yticks(range(len(cols)), cols, fontsize=5.5)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Pearson correlation")
cb.outline.set_visible(False)
ps.save(fig, FIG_DIR / "fig_correlation.pdf")

pd.Series(facts).to_csv(RESULTS_DIR / "eda_facts.csv", header=["value"])
print(f"\nSaved facts to {RESULTS_DIR / 'eda_facts.csv'}")

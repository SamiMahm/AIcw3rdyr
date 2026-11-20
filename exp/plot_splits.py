
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import plot_style as ps
from config import FIG_DIR, RESULTS_DIR

ps.apply()

runs = pd.read_csv(RESULTS_DIR / "splits_runs.csv")

SCHEMES = ["random", "house", "temporal", "spatial"]
LABELS = ["Random\n(paper)", "Grouped\nby house", "Temporal",
          "Spatial\n(unseen areas)"]
MODELS = [("Random Forest", ps.ORANGE), ("XGBoost", ps.BLUE)]
OFFSET = 0.17          # horizontal shift so the two models sit side by side

fig, ax = plt.subplots(figsize=(ps.COL_W, 2.2))
x = np.arange(len(SCHEMES))

for i, (model, colour) in enumerate(MODELS):
    shift = (i - 0.5) * 2 * OFFSET          # RF left, XGBoost right
    sub = runs[runs["model"] == model]
    for j, scheme in enumerate(SCHEMES):
        vals = sub.loc[sub["scheme"] == scheme, "MAPE"].values
        # one dot per seed
        ax.scatter(np.full(len(vals), x[j] + shift), vals, s=10,
                   color=colour, alpha=0.6, linewidths=0, zorder=3)
        # mean as a short horizontal bar
        ax.hlines(vals.mean(), x[j] + shift - 0.11, x[j] + shift + 0.11,
                  color=colour, linewidth=2, zorder=4,
                  label=model if j == 0 else None)

# Reference line: XGBoost mean under the paper's random split
ref = runs.query("model == 'XGBoost' and scheme == 'random'")["MAPE"].mean()
ax.axhline(ref, color=ps.GREY, linewidth=0.6, linestyle=(0, (3, 3)),
           zorder=1)
ax.text(x[-1] + 0.45, ref, "XGBoost,\nrandom split", fontsize=6,
        color=ps.GREY, va="center", ha="left")

ax.set_xticks(x, LABELS)
ax.set_xlim(-0.5, len(SCHEMES) - 0.5)
ax.set_ylabel("MAPE (%)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", frameon=False, handlelength=1.2)

ps.save(fig, FIG_DIR / "fig_splits_mape.pdf")

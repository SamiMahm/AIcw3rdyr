
import pandas as pd

import config as C

SCHEMES = ["random", "house", "temporal", "spatial"]
HEAD = ["Random", "House", "Temporal", "Spatial"]

ml = pd.read_csv(C.RESULTS_DIR / "splits_summary.csv")
sym = pd.read_csv(C.RESULTS_DIR / "symbolic_summary.csv")
hyb = pd.read_csv(C.RESULTS_DIR / "hybrid_summary.csv")
allm = pd.concat([ml, sym, hyb], ignore_index=True)


ROWS = [
    ("Median", "Median baseline", "Baselines"),
    ("Linear regression", "Linear Regression", "Baselines"),
    ("Zipcode \\$/sqft", "Rules: zipcode $/sqft only", "Symbolic"),
    ("+ adjustment grid", "Rules: + adjustment grid", "Symbolic"),
    ("Random Forest", "Random Forest", "ML"),
    ("XGBoost", "XGBoost", "ML"),
    ("XGBoost, log target", "XGBoost (log target)", "ML"),
    ("Rule as feature", "Hybrid: rule as feature", "Hybrid"),
    ("Residual", "Hybrid: residual", "Hybrid"),
]

cell = allm.set_index(["model", "scheme"])
best = {s: min(cell.loc[(m, s), "MAPE_mean"] for _, m, _ in ROWS)
        for s in SCHEMES}

lines = [r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "r" * len(SCHEMES) + "}",
         r"\hline",
         "Model & " + " & ".join(HEAD) + r" \\",
         r"\hline"]
group = None
for label, model, g in ROWS:
    if g != group:
        if group is not None:
            lines.append(r"\hline")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{g}}}}} \\")
        group = g
    cells = []
    for s in SCHEMES:
        m, sd = cell.loc[(model, s), ["MAPE_mean", "MAPE_std"]]
        txt = f"{m:.1f}$\\pm${sd:.1f}"
        cells.append(rf"\textbf{{{txt}}}" if abs(m - best[s]) < 1e-9 else txt)
    lines.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
lines += [r"\hline", r"\end{tabular}"]

out = C.RESULTS_DIR / "table_main_mape.tex"
out.write_text("\n".join(lines) + "\n")
print(out.read_text())
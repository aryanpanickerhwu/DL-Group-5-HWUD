import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

SETS = ["all", "metadata", "no_artist", "audio"]
SET_LABEL = {"all": "All", "metadata": "Metadata", "no_artist": "No artist", "audio": "Audio"}
SHORT = {"MLP (regression)": "MLP", "MLP classifier": "MLP classifier",
         "MLP regression, cut into classes": "MLP regr. (cut)", "Logistic Regression": "Log. Reg.",
         "Random Forest": "Rand. Forest", "Decision Tree": "Dec. Tree", "Naive Bayes": "Naive Bayes"}

def panel(ax, csv, col, ylabel, title):
    t = pd.read_csv(csv)
    pv = t.pivot(index="model", columns="features", values=col).reindex(columns=SETS)
    pv = pv.sort_values("all", ascending=False)
    pv.index = [SHORT.get(m, m) for m in pv.index]
    pv.columns = [SET_LABEL[c] for c in pv.columns]
    pv.plot.bar(ax=ax, width=0.8, legend=False)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_xlabel("")
    ax.set_xticklabels(pv.index, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

# ---------- Figure A1: MLP vs R3 models by feature set (validation) ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
panel(ax[0], "final_comparison_validation_regression.csv", "R2", "Validation R²", "(a) Regression")
panel(ax[1], "final_comparison_validation_classification.csv", "macro_F1", "Validation macro-F1", "(b) Classification")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, title="Feature set", loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("fig_r4_vs_r3_feature_sets_v2.png", dpi=200); plt.close(fig)

# ---------- Figure A2: confusion matrices of the MLP classifier (validation) ----------
CLASSES = ["Low", "Medium", "High"]
names = {"all": "All features", "metadata": "Metadata", "r3_no_artist": "No artist", "r3_audio": "Audio"}
fig, ax = plt.subplots(1, 4, figsize=(14, 3.7))
for a, (key, title) in zip(ax, names.items()):
    d = pd.read_csv(f"mlpclf_val_predictions_{'r3_metadata' if key == 'metadata' else key}.csv")
    ConfusionMatrixDisplay.from_predictions(d["y_true"], d["y_pred"], labels=CLASSES, normalize="true",
                                            values_format=".2f", cmap="Blues", ax=ax[list(names).index(key)] if False else a,
                                            colorbar=False)
    a.set_title(title)
fig.tight_layout(); fig.savefig("fig_r4_confusion_matrices.png", dpi=200); plt.close(fig)
print("saved fig_r4_vs_r3_feature_sets_v2.png and fig_r4_confusion_matrices.png")
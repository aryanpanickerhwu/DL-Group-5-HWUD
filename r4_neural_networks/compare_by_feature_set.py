import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score)

# ---------- where are the R3 feature-group tables? ----------
def find_r3():
    for d in ["results", "../results", "."]:
        if os.path.exists(os.path.join(d, "r3_reg_feature_groups.csv")):
            return d
    raise SystemExit("Could not find r3_reg_feature_groups.csv. Copy it and r3_cls_feature_groups.csv "
                     "into this folder (or a 'results' folder next to it) and run again.")
R3 = find_r3()

CLASSES = ["Low", "Medium", "High"]
SETS = ["all", "no_artist", "metadata", "audio"]
REG_FILE = {"all": "with_artist", "no_artist": "r3_no_artist", "metadata": "r3_metadata", "audio": "r3_audio"}
CLF_FILE = {"all": "all", "no_artist": "r3_no_artist", "metadata": "r3_metadata", "audio": "r3_audio"}

def to_class(x):                      # Low 0-40, Medium 41-60, High 61-100
    return pd.cut(pd.Series(x), [-np.inf, 40.5, 60.5, np.inf], labels=CLASSES)

def cls_metrics(yc, pc):
    f1 = f1_score(yc, pc, labels=CLASSES, average=None, zero_division=0)
    return {"accuracy": accuracy_score(yc, pc), "macro_F1": f1.mean(),
            "Low_F1": f1[0], "Medium_F1": f1[1], "High_F1": f1[2],
            "High_recall": recall_score(yc, pc, labels=["High"], average=None, zero_division=0)[0]}

# ---------- R3 tables ----------
reg3 = pd.read_csv(os.path.join(R3, "r3_reg_feature_groups.csv"))[["features", "model", "RMSE", "MAE", "R2"]]
cls3 = pd.read_csv(os.path.join(R3, "r3_cls_feature_groups.csv"))[
    ["features", "model", "accuracy", "macro_F1", "Low_F1", "Medium_F1", "High_F1", "High_recall"]]

# ---------- MLP rows (validation predictions saved by the two MLP scripts) ----------
reg_rows, cls_rows = [], []
for fs in SETS:
    p = f"mlp_val_predictions_{REG_FILE[fs]}.csv"            # regression MLP
    if os.path.exists(p):
        d = pd.read_csv(p); y, pr = d["y_true"].values, d["y_pred"].values
        reg_rows.append({"features": fs, "model": "MLP (regression)",
                         "RMSE": mean_squared_error(y, pr) ** 0.5,
                         "MAE": mean_absolute_error(y, pr), "R2": r2_score(y, pr)})
        cls_rows.append({"features": fs, "model": "MLP regression, cut into classes",
                         **cls_metrics(to_class(y), to_class(pr))})
    else:
        print("missing:", p)
    p = f"mlpclf_val_predictions_{CLF_FILE[fs]}.csv"         # classification MLP
    if os.path.exists(p):
        d = pd.read_csv(p)
        cls_rows.append({"features": fs, "model": "MLP classifier",
                         **cls_metrics(pd.Series(pd.Categorical(d["y_true"], CLASSES)),
                                       pd.Series(pd.Categorical(d["y_pred"], CLASSES)))})
    else:
        print("missing:", p)

reg_all = pd.concat([reg3, pd.DataFrame(reg_rows)], ignore_index=True)
cls_all = pd.concat([cls3, pd.DataFrame(cls_rows)], ignore_index=True)
reg_all.to_csv("final_comparison_validation_regression.csv", index=False)
cls_all.to_csv("final_comparison_validation_classification.csv", index=False)

pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
def pivot(t, col):
    pv = t.pivot(index="model", columns="features", values=col).reindex(columns=SETS)
    return pv.sort_values("all", ascending=(col in ("RMSE", "MAE")))
for col in ["R2", "RMSE", "MAE"]:
    print(f"\nREGRESSION validation {col} by feature set"); print(pivot(reg_all, col).to_string())
for col in ["macro_F1", "accuracy", "High_recall"]:
    print(f"\nCLASSIFICATION validation {col} by feature set"); print(pivot(cls_all, col).to_string())

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
pivot(reg_all, "R2").plot.bar(ax=ax[0], rot=30); ax[0].set(ylabel="Validation R²", title="Regression: R² by feature set")
pivot(cls_all, "macro_F1").plot.bar(ax=ax[1], rot=30); ax[1].set(ylabel="Validation macro-F1", title="Classification: macro-F1 by feature set")
for a in ax: a.legend(title="features", fontsize=8)
plt.tight_layout(); plt.savefig("fig_r4_vs_r3_feature_sets.png", dpi=150)
print("\nsaved final_comparison_validation_regression.csv, final_comparison_validation_classification.csv, "
      "fig_r4_vs_r3_feature_sets.png")
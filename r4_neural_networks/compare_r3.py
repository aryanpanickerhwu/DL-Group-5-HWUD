import os
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score)

# ---------- where are the R3 validation tables? ----------
def find_r3():
    for d in ["results", "../results", "."]:
        if os.path.exists(os.path.join(d, "r3_reg_validation.csv")):
            return d
    raise SystemExit("Could not find r3_reg_validation.csv. Copy it and r3_cls_validation.csv "
                     "into this folder (or a 'results' folder next to it) and run again.")
R3 = find_r3()

CLASSES = ["Low", "Medium", "High"]
def to_class(x):                      # Low 0-40, Medium 41-60, High 61-100 (cut at 40.5 and 60.5)
    return pd.cut(pd.Series(x), [-np.inf, 40.5, 60.5, np.inf], labels=CLASSES)

# ---------- MLP validation predictions (saved by mlp_keras.py) ----------
versions = {"MLP (all features)": "with_artist",
            "MLP (selected 3 features)": "selected_3",
            "MLP (2 artist columns removed, genres kept)": "without_artist"}
reg_mlp, cls_mlp = [], []
for label, name in versions.items():
    path = f"mlp_val_predictions_{name}.csv"
    if not os.path.exists(path):
        print("missing:", path); continue
    d = pd.read_csv(path)
    y, p = d["y_true"].values, d["y_pred"].values
    reg_mlp.append({"model": label, "RMSE": mean_squared_error(y, p) ** 0.5,
                    "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p)})
    yc, pc = to_class(y), to_class(p)
    f1 = f1_score(yc, pc, labels=CLASSES, average=None)
    cls_mlp.append({"model": label, "accuracy": accuracy_score(yc, pc),
                    "macro_F1": f1.mean(), "Low_F1": f1[0], "Medium_F1": f1[1], "High_F1": f1[2],
                    "High_recall": recall_score(yc, pc, labels=["High"], average=None)[0]})
reg_mlp, cls_mlp = pd.DataFrame(reg_mlp), pd.DataFrame(cls_mlp)

# ---------- R3 tables (validation, all features) ----------
reg3 = pd.read_csv(os.path.join(R3, "r3_reg_validation.csv"))[["model", "RMSE", "MAE", "R2"]]
cls3 = pd.read_csv(os.path.join(R3, "r3_cls_validation.csv"))[
    ["model", "accuracy", "macro_F1", "Low_F1", "Medium_F1", "High_F1", "High_recall"]]

pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
main_label = "MLP (all features)"
reg_main = pd.concat([reg3, reg_mlp[reg_mlp.model == main_label]]).sort_values("RMSE")
cls_main = pd.concat([cls3, cls_mlp[cls_mlp.model == main_label]]).sort_values("macro_F1", ascending=False)

print("\nREGRESSION, validation set, all features (lower RMSE/MAE is better)")
print(reg_main.to_string(index=False))
print("\nCLASSIFICATION, validation set, all features. The MLP row comes from its regression "
      "predictions cut into Low/Medium/High")
print(cls_main.to_string(index=False))
print("\nOther MLP feature sets (not directly comparable with the R3 tables above):")
print(reg_mlp.to_string(index=False))
print(cls_mlp.to_string(index=False))

reg_main.to_csv("r4_vs_r3_validation_regression.csv", index=False)
cls_main.to_csv("r4_vs_r3_validation_classification.csv", index=False)
print("\nsaved r4_vs_r3_validation_regression.csv and r4_vs_r3_validation_classification.csv")
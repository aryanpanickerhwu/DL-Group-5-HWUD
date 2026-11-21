import os
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score)

def find_r3():
    for d in ["results", "../results", "."]:
        if os.path.exists(os.path.join(d, "r3_reg_TEST.csv")):
            return d
    raise SystemExit("Could not find r3_reg_TEST.csv / r3_cls_TEST.csv (the output of R3's final_test()). "
                     "Copy them into this folder or a 'results' folder and run again.")
R3 = find_r3()
CLASSES = ["Low", "Medium", "High"]
def to_class(x):                      # Low 0-40, Medium 41-60, High 61-100
    return pd.cut(pd.Series(x), [-np.inf, 40.5, 60.5, np.inf], labels=CLASSES)
def cls_metrics(yc, pc):
    f1 = f1_score(yc, pc, labels=CLASSES, average=None, zero_division=0)
    return {"accuracy": accuracy_score(yc, pc), "macro_F1": f1.mean(), "Low_F1": f1[0],
            "Medium_F1": f1[1], "High_F1": f1[2],
            "High_recall": recall_score(yc, pc, labels=["High"], average=None, zero_division=0)[0]}

REG_COLS = ["model", "RMSE", "MAE", "R2"]
CLS_COLS = ["model", "accuracy", "macro_F1", "Low_F1", "Medium_F1", "High_F1", "High_recall"]
reg3 = pd.read_csv(os.path.join(R3, "r3_reg_TEST.csv"))
cls3 = pd.read_csv(os.path.join(R3, "r3_cls_TEST.csv"))
reg3 = reg3[[c for c in REG_COLS if c in reg3.columns]]
cls3 = cls3[[c for c in CLS_COLS if c in cls3.columns]]

# ---------- MLP rows (test predictions saved by the MLP scripts; all features) ----------
d = pd.read_csv("mlp_test_predictions_with_artist.csv"); y, p = d["y_true"].values, d["y_pred"].values
reg_mlp = pd.DataFrame([{"model": "MLP (regression)", "RMSE": mean_squared_error(y, p) ** 0.5,
                         "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p)}])
cut = pd.DataFrame([{"model": "MLP regression, cut into classes", **cls_metrics(to_class(y), to_class(p))}])
d = pd.read_csv("mlpclf_test_predictions_all.csv")
clf = pd.DataFrame([{"model": "MLP classifier",
                     **cls_metrics(pd.Series(pd.Categorical(d["y_true"], CLASSES)),
                                   pd.Series(pd.Categorical(d["y_pred"], CLASSES)))}])

reg_all = pd.concat([reg3, reg_mlp]).sort_values("RMSE")
cls_all = pd.concat([cls3, cut, clf]).sort_values("macro_F1", ascending=False)
pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
print("\nREGRESSION, test set, all features"); print(reg_all.to_string(index=False))
print("\nCLASSIFICATION, test set, all features"); print(cls_all.to_string(index=False))
reg_all.to_csv("final_comparison_test_regression.csv", index=False)
cls_all.to_csv("final_comparison_test_classification.csv", index=False)
print("\nsaved final_comparison_test_regression.csv and final_comparison_test_classification.csv")
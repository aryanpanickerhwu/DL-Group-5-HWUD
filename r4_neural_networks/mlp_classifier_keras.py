import json, os, random, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quieter TensorFlow output
import numpy as np, pandas as pd, keras
from keras import layers, regularizers
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

SEED, N_FOLDS = 42, 10
N_CONFIGS = 6                       # settings tried per version; lower it (e.g. 3) if it is too slow
# usage:  py mlp_classifier_keras.py all r3_no_artist r3_audio r3_metadata        (tune + validation)
#         py mlp_classifier_keras.py all r3_no_artist r3_audio r3_metadata test   (final test, once)
RUN_TEST = "test" in sys.argv[1:]
RUN_VERSIONS = [a for a in sys.argv[1:] if a != "test"] or ["all"]
CLASSES = ["Low", "Medium", "High"]  # Low 0-40, Medium 41-60, High 61-100

# ---------- data ----------
df = pd.read_csv("spotgentrack_clean.csv")
df["split"] = df["split"].replace({"validation": "val"})
assert set(df["split"].unique()) == {"train", "val", "test"}, df["split"].unique()
assert set(df["pop_class"].unique()) == set(CLASSES), df["pop_class"].unique()
df["y"] = df["pop_class"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)

DROP = ["id", "split", "popularity", "pop_class", "song_id", "y"]   # none of these is ever a feature
ALL_FEATURES = [c for c in df.columns if c not in DROP]
VERSIONS = {"all": ALL_FEATURES}

# feature sets as defined by the R3 notebooks
sys.path.insert(0, os.getcwd())
try:
    import r3_common as rc
    R3G = rc.feature_groups(df)
except ImportError:
    audio_names = ["acousticness", "danceability", "energy", "instrumentalness", "liveness",
                   "loudness", "mode", "speechiness", "tempo", "valence"]
    audio = [c for c in ALL_FEATURES if c in audio_names or c.startswith(("key_", "time_signature_"))]
    R3G = {"audio": audio,
           "metadata": [c for c in ALL_FEATURES if c not in audio],
           "no_artist": [c for c in ALL_FEATURES
                         if c not in ("artist_popularity", "log_followers")
                         and not c.startswith("genre_broad_")]}
for k in ("no_artist", "audio", "metadata"):
    VERSIONS["r3_" + k] = [c for c in R3G[k] if c in ALL_FEATURES]
print({k: len(v) for k, v in VERSIONS.items()}, "(expect 49 / 34 / 26 / 23)")
assert all(v in VERSIONS for v in RUN_VERSIONS), f"unknown version in {RUN_VERSIONS}; choose from {list(VERSIONS)}"
print("versions to run:", RUN_VERSIONS, "| final test:", RUN_TEST)

train, val, test = (df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
print("train/val/test:", len(train), len(val), len(test))
assert not set(train.song_id) & set(val.song_id) and not set(train.song_id) & set(test.song_id) \
       and not set(val.song_id) & set(test.song_id), "a song appears in more than one split"
yva = val["y"].values


# ---------- helpers ----------
def clf_metrics(y, p):
    pr, rc_, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    out = {"accuracy": float(accuracy_score(y, p)), "macro_F1": float(f1.mean()),
           "weighted_F1": float(f1_score(y, p, average="weighted", zero_division=0))}
    for i, c in enumerate(CLASSES):
        out[f"{c}_precision"], out[f"{c}_recall"], out[f"{c}_F1"] = float(pr[i]), float(rc_[i]), float(f1[i])
    return out


def prepare(feats, fit_df, *others):
    """The scaler is fitted on fit_df only (no leakage)."""
    cont = [c for c in feats if df[c].nunique() > 2]
    sc = StandardScaler().fit(fit_df[cont])
    def tx(d):
        X = d[feats].astype("float32").copy()
        X[cont] = sc.transform(d[cont]).astype("float32")
        return X.values
    return tx(fit_df), [tx(d) for d in others]


def build_model(n_in, cfg):
    keras.utils.set_random_seed(SEED)
    reg = regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    m = keras.Sequential([keras.Input(shape=(n_in,))])
    for h in cfg["hidden"]:
        m.add(layers.Dense(h, activation="relu", kernel_regularizer=reg))
        if cfg["dropout"] > 0: m.add(layers.Dropout(cfg["dropout"]))
    m.add(layers.Dense(3, activation="softmax"))
    m.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss="sparse_categorical_crossentropy")
    return m


class MacroF1Stop(keras.callbacks.Callback):
    """Early stopping on the validation set's macro-F1; the best weights are restored at the end."""
    def __init__(self, Xva, yva, patience=10):
        super().__init__(); self.Xva, self.yva, self.patience = Xva, yva, patience
    def on_train_begin(self, logs=None):
        self.best, self.wait, self.best_w = -1.0, 0, None
    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.Xva, batch_size=4096, verbose=0).argmax(1)
        f = f1_score(self.yva, p, average="macro")
        if f > self.best + 1e-4:
            self.best, self.wait, self.best_w = f, 0, self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience: self.model.stop_training = True
    def on_train_end(self, logs=None):
        if self.best_w is not None: self.model.set_weights(self.best_w)


def fit_model(Xtr, ytr, Xva, yva_, cfg):
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(3), y=ytr)))
    model = build_model(Xtr.shape[1], cfg)
    model.fit(Xtr, ytr, epochs=100, batch_size=cfg["bs"], class_weight=cw,
              callbacks=[MacroF1Stop(Xva, yva_)], verbose=0)
    return model


def predict(model, X):
    return model.predict(X, batch_size=4096, verbose=0).argmax(1)


# ---------- 10-fold stratified group CV on TRAIN, scored by macro-F1 ----------
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(cv.split(train, train["pop_class"], groups=train["song_id"]))

def cv_score(feats, cfg):
    scores = []
    for tr_i, ho_i in folds:
        tr_df, ho_df = train.iloc[tr_i], train.iloc[ho_i]
        Xtr, (Xho, Xva) = prepare(feats, tr_df, ho_df, val)
        # early stopping uses the validation set (team protocol); the held-out fold is never seen
        model = fit_model(Xtr, tr_df["y"].values, Xva, yva, cfg)
        scores.append(f1_score(ho_df["y"].values, predict(model, Xho), average="macro"))
        keras.backend.clear_session()
    return float(np.mean(scores)), float(np.std(scores))

grid = [{"hidden": h, "dropout": d, "lr": lr, "l2": l2, "bs": 512}
        for h in [[64], [128, 64], [256, 128]] for d in [0.0, 0.2, 0.4]
        for lr in [1e-3, 3e-4] for l2 in [0.0, 1e-4]]
configs = random.Random(SEED).sample(grid, N_CONFIGS)

def tune(name):
    path = f"mlpclf_best_{name}.json"          # delete this file to redo the search
    if os.path.exists(path):
        return json.load(open(path))["cfg"]
    rows = []
    for k, cfg in enumerate(configs, 1):
        m, s = cv_score(VERSIONS[name], cfg)
        rows.append({**cfg, "hidden": str(cfg["hidden"]), "cv_macro_F1": m, "cv_sd": s})
        print(f"[{name}] {k}/{N_CONFIGS} {cfg} -> CV macro-F1 {m:.3f} ± {s:.3f}", flush=True)
    pd.DataFrame(rows).sort_values("cv_macro_F1", ascending=False).to_csv(f"mlpclf_cv_{name}.csv", index=False)
    best = configs[int(np.argmax([r["cv_macro_F1"] for r in rows]))]
    json.dump({"cfg": best}, open(path, "w"))
    return best

def final_fit(name, cfg, include_test):
    """Refit on the full train set, early stop on validation."""
    Xtr, others = prepare(VERSIONS[name], train, val, *([test] if include_test else []))
    model = fit_model(Xtr, train["y"].values, others[0], yva, cfg)
    return predict(model, others[0]), (predict(model, others[1]) if include_test else None)


# ---------- run ----------
SUFFIX = "_".join(RUN_VERSIONS)
majority = int(train["y"].value_counts().idxmax())
if not RUN_TEST:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    out = [{"model": "majority_baseline", **clf_metrics(yva, np.full(len(val), majority))}]
    for name in RUN_VERSIONS:
        cfg = tune(name)
        pv, _ = final_fit(name, cfg, include_test=False)
        out.append({"model": f"mlp_clf_{name}", **clf_metrics(yva, pv)})
        pd.DataFrame({"id": val["id"], "y_true": [CLASSES[i] for i in yva],
                      "y_pred": [CLASSES[i] for i in pv]}).to_csv(
            f"mlpclf_val_predictions_{name}.csv", index=False)
        fig, ax = plt.subplots(figsize=(4.4, 3.9))
        ConfusionMatrixDisplay.from_predictions(yva, pv, labels=[0, 1, 2], display_labels=CLASSES,
                                                normalize="true", values_format=".2f", cmap="Blues",
                                                ax=ax, colorbar=False)
        ax.set_title(f"MLP classifier ({name}), validation")
        fig.tight_layout(); fig.savefig(f"mlpclf_confusion_{name}.png", dpi=150); plt.close(fig)
    res = pd.DataFrame(out)
    print("\nVALIDATION RESULTS")
    print(res[["model", "accuracy", "macro_F1", "Low_F1", "Medium_F1", "High_F1", "High_recall"]].to_string(index=False))
    res.to_csv(f"mlpclf_validation_results_{SUFFIX}.csv", index=False)
else:
    yte = test["y"].values
    all_res = {"majority_baseline_test": clf_metrics(yte, np.full(len(test), majority))}
    for name in RUN_VERSIONS:
        cfg = tune(name)                      # loads the saved best settings, no new search
        _, pt = final_fit(name, cfg, include_test=True)
        all_res[f"mlp_clf_{name}_test"] = {"config": cfg, **clf_metrics(yte, pt)}
        pd.DataFrame({"id": test["id"], "y_true": [CLASSES[i] for i in yte],
                      "y_pred": [CLASSES[i] for i in pt]}).to_csv(
            f"mlpclf_test_predictions_{name}.csv", index=False)
    print(json.dumps(all_res, indent=2))
    json.dump(all_res, open(f"mlpclf_test_results_{SUFFIX}.json", "w"), indent=2)
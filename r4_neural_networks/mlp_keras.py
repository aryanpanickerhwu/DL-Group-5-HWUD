import json, os, random
import numpy as np, pandas as pd
import keras
import sys
from keras import layers, regularizers
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED, N_FOLDS = 42, 10
N_CONFIGS = 6                    # settings tried per version; lower it (e.g. 3) if it is too slow
RUN_TEST = "test" in sys.argv[1:]               # keep False until the team agrees which version to report
TEST_VERSION = "with_artist"     # only used when RUN_TEST = True
RUN_VERSIONS =  [a for a in sys.argv[1:] if a != "test"] or ["r3_no_artist"] # only the new version; the other two are already finished

# ---------- data ----------
df = pd.read_csv("spotgentrack_clean.csv")
df["split"] = df["split"].replace({"validation": "val"})
assert set(df["split"].unique()) == {"train", "val", "test"}, df["split"].unique()
assert "song_id" in df.columns, "re-download spotgentrack_clean.csv from GitHub"
TARGET = "popularity"
DROP = ["id", "split", "popularity", "pop_class", "song_id"]   # song_id = grouping only
ALL_FEATURES = [c for c in df.columns if c not in DROP]
ARTIST = ["artist_popularity", "log_followers"]
VERSIONS = {"with_artist": ALL_FEATURES,
            "without_artist": [c for c in ALL_FEATURES if c not in ARTIST]}
# third version: the features chosen by the team's feature selection
SEL = [s.strip() for s in open("selected_features.txt").read().splitlines() if s.strip()]
VERSIONS["selected_3"] = [c for c in ALL_FEATURES
                          if c in SEL or any(c.startswith(s + "_") for s in SEL)]
print("selected_3 columns:", VERSIONS["selected_3"])
# feature sets as defined by the R3 notebooks          
import sys; sys.path.insert(0, os.getcwd())
try:
    import r3_common as rc
    R3G = rc.feature_groups(df)
except ImportError:
    ...                                                 
for k in ("no_artist", "audio", "metadata"):
    VERSIONS["r3_" + k] = [c for c in R3G[k] if c in ALL_FEATURES]
print({k: len(v) for k, v in VERSIONS.items()})        
print({k: len(v) for k, v in VERSIONS.items()}, "(earlier file: 49 features with artist)")

train, val, test = (df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
print("train/val/test:", len(train), len(val), len(test))
assert not set(train.song_id) & set(val.song_id) and not set(train.song_id) & set(test.song_id) \
       and not set(val.song_id) & set(test.song_id), "a song appears in more than one split"
yva = val[TARGET].values

# ---------- helpers ----------
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
def metrics(a, b):
    return {"MAE": float(mean_absolute_error(a, b)), "RMSE": rmse(a, b), "R2": float(r2_score(a, b))}

def prepare(feats, fit_df, *others):
    """Scaler and target stats are fitted on fit_df only (no leakage)."""
    cont = [c for c in feats if df[c].nunique() > 2]
    sc = StandardScaler().fit(fit_df[cont])
    def tx(d):
        X = d[feats].astype("float32").copy()
        X[cont] = sc.transform(d[cont]).astype("float32")
        return X.values
    return tx(fit_df), [tx(d) for d in others], fit_df[TARGET].mean(), fit_df[TARGET].std()

def build_model(n_in, cfg):
    keras.utils.set_random_seed(SEED)
    reg = regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    m = keras.Sequential([keras.Input(shape=(n_in,))])
    for h in cfg["hidden"]:
        m.add(layers.Dense(h, activation="relu", kernel_regularizer=reg))
        if cfg["dropout"] > 0: m.add(layers.Dropout(cfg["dropout"]))
    m.add(layers.Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss="mse",
              metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    return m

def fit_model(Xtr, ytr, Xva, yva_s, cfg):
    model = build_model(Xtr.shape[1], cfg)
    es = keras.callbacks.EarlyStopping(monitor="val_rmse", mode="min", patience=10,
                                       restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xva, yva_s), epochs=100,
              batch_size=cfg["bs"], callbacks=[es], verbose=0)
    return model

# ---------- 10-fold stratified group CV on TRAIN ----------
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(cv.split(train, train["pop_class"], groups=train["song_id"]))

def cv_score(feats, cfg):
    scores = []
    for tr_i, ho_i in folds:
        tr_df, ho_df = train.iloc[tr_i], train.iloc[ho_i]
        Xtr, (Xho, Xva), mu, sd = prepare(feats, tr_df, ho_df, val)
        # early stopping uses the validation set (team protocol); the held-out fold is never seen
        model = fit_model(Xtr, (tr_df[TARGET].values - mu) / sd, Xva, (yva - mu) / sd, cfg)
        p = model.predict(Xho, verbose=0).ravel() * sd + mu
        scores.append(rmse(ho_df[TARGET].values, p))
        keras.backend.clear_session()
    return float(np.mean(scores)), float(np.std(scores))

grid = [{"hidden": h, "dropout": d, "lr": lr, "l2": l2, "bs": 512}
        for h in [[64], [128, 64], [256, 128]] for d in [0.0, 0.2, 0.4]
        for lr in [1e-3, 3e-4] for l2 in [0.0, 1e-4]]
configs = random.Random(SEED).sample(grid, N_CONFIGS)

def tune(name):
    path = f"mlp_best_{name}.json"            # delete this file to redo the search
    if os.path.exists(path):
        return json.load(open(path))["cfg"]
    rows = []
    for k, cfg in enumerate(configs, 1):
        m, s = cv_score(VERSIONS[name], cfg)
        rows.append({**cfg, "hidden": str(cfg["hidden"]), "cv_rmse": m, "cv_sd": s})
        print(f"[{name}] {k}/{N_CONFIGS} {cfg} -> CV RMSE {m:.3f} ± {s:.3f}")
    pd.DataFrame(rows).sort_values("cv_rmse").to_csv(f"mlp_cv_{name}.csv", index=False)
    best = configs[int(np.argmin([r["cv_rmse"] for r in rows]))]
    json.dump({"cfg": best}, open(path, "w"))
    return best

def final_fit(name, cfg, include_test):
    """Refit on the full train set, early stop on validation."""
    Xtr, others, mu, sd = prepare(VERSIONS[name], train, val, *([test] if include_test else []))
    model = fit_model(Xtr, (train[TARGET].values - mu) / sd, others[0], (yva - mu) / sd, cfg)
    pred = lambda X: model.predict(X, verbose=0).ravel() * sd + mu
    return pred(others[0]), (pred(others[1]) if include_test else None)

# ---------- run ----------
SUFFIX = "_".join(RUN_VERSIONS)
if not RUN_TEST:
    out = [{"model": "mean_baseline", **metrics(yva, np.full(len(val), train[TARGET].mean()))}]
    for name in RUN_VERSIONS:
        cfg = tune(name)
        pv, _ = final_fit(name, cfg, include_test=False)
        out.append({"model": f"mlp_{name}", **metrics(yva, pv)})
        pd.DataFrame({"id": val["id"], "y_true": yva, "y_pred": pv}).to_csv(
            f"mlp_val_predictions_{name}.csv", index=False)
    res = pd.DataFrame(out); print("\nVALIDATION RESULTS"); print(res.to_string(index=False))
    res.to_csv(f"mlp_validation_results_{SUFFIX}.csv", index=False)
else:
    yte = test[TARGET].values
    all_res = {"mean_baseline_test": metrics(yte, np.full(len(test), train[TARGET].mean()))}
    for name in RUN_VERSIONS:
        cfg = tune(name)                      # loads the saved best settings, no new search
        _, pt = final_fit(name, cfg, include_test=True)
        all_res[f"mlp_{name}_test"] = {"config": cfg, **metrics(yte, pt)}
        pd.DataFrame({"id": test["id"], "y_true": yte, "y_pred": pt}).to_csv(
            f"mlp_test_predictions_{name}.csv", index=False)
    print(json.dumps(all_res, indent=2))
    json.dump(all_res, open(f"mlp_test_results_{SUFFIX}.json", "w"), indent=2)
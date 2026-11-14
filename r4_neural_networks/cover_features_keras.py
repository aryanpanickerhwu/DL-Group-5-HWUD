import sys
import numpy as np, pandas as pd, keras
from keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

VERSION = sys.argv[1] if len(sys.argv) > 1 else "with_artist"
assert VERSION in ("with_artist", "without_artist"), "run: py cover_features_keras.py with_artist  (or without_artist)"
SEEDS, KS, IMG_KEY = [0, 1, 2], [8, 16, 32], "X_visual"

# ---------- load matched tracks ----------
z = np.load("matched_visual_features.npz")
Xv = z[IMG_KEY]
m = pd.read_csv("matched_ids.csv", usecols=["id", "msd_track_id", "split"])
assert len(m) == len(Xv)
msdi = pd.read_csv("MSD-I_dataset.tsv", sep="\t")[["msd_track_id", "msd_artist_id"]]
m = m.merge(msdi, on="msd_track_id", how="left")
assert len(m) == len(Xv) and m["msd_artist_id"].notna().all()

clean = pd.read_csv("spotgentrack_clean.csv")
clean["split"] = clean["split"].replace({"validation": "val"})
DROP = ["id", "split", "popularity", "pop_class", "song_id"]       # song_id is never a feature
feature_cols = [c for c in clean.columns if c not in DROP]
if VERSION == "without_artist":
    feature_cols = [c for c in feature_cols if c not in ("artist_popularity", "log_followers")]
cont_cols = [c for c in feature_cols if clean[c].nunique() > 2]
cl = clean.set_index("id").loc[m["id"]]
assert (cl["split"].values == m["split"].values).all(), "splits differ: re-run build_matched_features.py"
T = cl[feature_cols].astype("float32").reset_index(drop=True)
y = cl["popularity"].values.astype("float32")
groups = m["msd_artist_id"].values
print("version:", VERSION, "| features:", len(feature_cols), "| tracks:", len(m),
      "| splits:", m["split"].value_counts().to_dict())

# ---------- helpers ----------
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
def metrics(a, b): return {"MAE": mean_absolute_error(a, b), "RMSE": rmse(a, b), "R2": r2_score(a, b)}

def build(kind, k, tr, others):
    """Scalers and PCA are fitted on the training rows only."""
    sc = StandardScaler().fit(T.loc[tr, cont_cols])
    def tab(ix):
        X = T.loc[ix].copy(); X[cont_cols] = sc.transform(X[cont_cols])
        return X.values.astype("float32")
    ptr, po = [tab(tr)], [[tab(o)] for o in others]
    if kind == "both":
        s1 = StandardScaler().fit(Xv[tr])
        pca = PCA(n_components=k, random_state=0).fit(s1.transform(Xv[tr]))
        s2 = StandardScaler().fit(pca.transform(s1.transform(Xv[tr])))
        def img(ix): return s2.transform(pca.transform(s1.transform(Xv[ix]))).astype("float32")
        ptr.append(img(tr))
        for p, o in zip(po, others): p.append(img(o))
    return np.hstack(ptr), [np.hstack(p) for p in po]

def fit_predict(Xtr, ytr, Xva, yva, Xte, seed):
    keras.utils.set_random_seed(seed)
    mu, sd = float(ytr.mean()), float(ytr.std())
    model = keras.Sequential([
        keras.Input(shape=(Xtr.shape[1],)),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.3), layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
                  metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    es = keras.callbacks.EarlyStopping(monitor="val_rmse", mode="min", patience=30,
                                       restore_best_weights=True)
    model.fit(Xtr, (ytr - mu) / sd, validation_data=(Xva, (yva - mu) / sd),
              epochs=300, batch_size=32, callbacks=[es], verbose=0)
    p = model.predict(Xte, verbose=0).ravel() * sd + mu
    keras.backend.clear_session()
    return p

CONFIGS = [("mean_baseline", None, None), ("tabular", "tab", None)] \
        + [(f"tabular+image_k{k}", "both", k) for k in KS]

def run_configs(tr, va, te, seed):
    out = {}
    for name, kind, k in CONFIGS:
        if kind is None:
            out[name] = np.full(len(te), y[tr].mean())
        else:
            Xtr, (Xva, Xte) = build(kind, k, tr, [va, te])
            out[name] = fit_predict(Xtr, y[tr], Xva, y[va], Xte, seed)
    return out

def summarize(res):
    rows = []
    for name, lst in res.items():
        d = pd.DataFrame(lst)
        rows.append({"model": name, **{c: f"{d[c].mean():.3f} ± {d[c].std():.3f}" for c in d.columns}})
    return pd.DataFrame(rows)

# ---------- Evaluation 1: the team's split ----------
tr, va, te = (np.where(m["split"] == s)[0] for s in ["train", "val", "test"])
res = {n: [] for n, _, _ in CONFIGS}
for s in SEEDS:
    for name, p in run_configs(tr, va, te, s).items():
        res[name].append(metrics(y[te], p))
    print("team split: seed", s, "done")
r1 = summarize(res)
print("\nTEAM SPLIT (test n=%d), %s" % (len(te), VERSION)); print(r1.to_string(index=False))
r1.to_csv(f"keras_cover_results_team_split_{VERSION}.csv", index=False)

# ---------- Evaluation 2: 5-fold CV, all tracks of an artist stay in one fold ----------
def artist_folds(g, n_folds, seed):
    ug = np.array(sorted(set(g))); np.random.RandomState(seed).shuffle(ug)
    fo = {a: i % n_folds for i, a in enumerate(ug)}
    return np.array([fo[a] for a in g])

def inner_split(pool, seed):
    gp = groups[pool]; ug = np.array(sorted(set(gp))); np.random.RandomState(seed).shuffle(ug)
    vg = set(ug[: max(1, int(0.2 * len(ug)))]); isv = np.array([a in vg for a in gp])
    return pool[~isv], pool[isv]

res_cv = {n: [] for n, _, _ in CONFIGS}
for s in SEEDS:
    f = artist_folds(groups, 5, s)
    oof = {n: np.zeros(len(m)) for n, _, _ in CONFIGS}
    for k in range(5):
        te_k = np.where(f == k)[0]; pool = np.where(f != k)[0]
        tr_k, va_k = inner_split(pool, s)
        for name, p in run_configs(tr_k, va_k, te_k, s).items():
            oof[name][te_k] = p
    for name in oof: res_cv[name].append(metrics(y, oof[name]))
    print("grouped CV: seed", s, "done")
r2 = summarize(res_cv)
print("\nARTIST-GROUPED 5-FOLD CV (all %d tracks), %s" % (len(m), VERSION)); print(r2.to_string(index=False))
r2.to_csv(f"keras_cover_results_grouped_cv_{VERSION}.csv", index=False)
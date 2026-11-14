import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quieter TensorFlow output
import numpy as np, pandas as pd, keras
from keras import layers, regularizers
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

IMG = 128                    # covers are resized to 128 x 128
WEIGHTS = "imagenet"         # pretrained weights (downloaded once by Keras)
UNFREEZE = 20                # how many top layers of the CNN are fine-tuned
SEEDS = [0]                  # use [0, 1, 2] for more stable numbers (3x slower)


def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
def metrics(a, b): return {"MAE": mean_absolute_error(a, b), "RMSE": rmse(a, b), "R2": r2_score(a, b)}


def load_images(paths):
    arr = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((IMG, IMG))
        arr.append(np.asarray(im, dtype="float32"))     # pixel values 0..255
    return np.stack(arr)


# ---------- the CNN ----------
def build_model(n_tab, use_tab):
    base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights=WEIGHTS)
    base.trainable = False                                # stage 1: pretrained CNN frozen
    img_in = keras.Input(shape=(IMG, IMG, 3))
    x = layers.RandomFlip("horizontal")(img_in)           # augmentation, only active in training
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)     # MobileNetV2 expects inputs in [-1, 1]
    x = base(x, training=False)                           # keeps BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    inputs = [img_in]
    if use_tab:
        tab_in = keras.Input(shape=(n_tab,))
        x = layers.Concatenate()([x, tab_in])
        inputs.append(tab_in)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1)(x)
    return keras.Model(inputs, out), base


def fit_predict_cnn(Itr, Ttr, ytr, Iva, Tva, yva, Ite, Tte, seed, use_tab):
    keras.utils.set_random_seed(seed)
    mu, sd = float(ytr.mean()), float(ytr.std())
    model, base = build_model(Ttr.shape[1] if use_tab else 0, use_tab)
    pack = (lambda I, T: [I, T]) if use_tab else (lambda I, T: I)

    def run(lr, epochs, patience):
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse",
                      metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
        es = keras.callbacks.EarlyStopping(monitor="val_rmse", mode="min", patience=patience,
                                           restore_best_weights=True)
        model.fit(pack(Itr, Ttr), (ytr - mu) / sd,
                  validation_data=(pack(Iva, Tva), (yva - mu) / sd),
                  epochs=epochs, batch_size=32, callbacks=[es], verbose=0)

    run(1e-3, 40, 8)                                      # stage 1: train the new layers only
    base.trainable = True                                 # stage 2: fine-tune the top CNN layers
    for layer in base.layers[:-UNFREEZE]:
        layer.trainable = False
    for layer in base.layers[-UNFREEZE:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)
    run(1e-5, 20, 6)                                      # small learning rate for fine-tuning
    p = model.predict(pack(Ite, Tte), batch_size=64, verbose=0).ravel() * sd + mu
    keras.backend.clear_session()
    return p


# ---------- plain Keras MLP on tabular features (same data, for a fair comparison) ----------
def fit_predict_mlp(Xtr, ytr, Xva, yva, Xte, seed):
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


def artist_folds(g, n_folds, seed):
    ug = np.array(sorted(set(g))); np.random.RandomState(seed).shuffle(ug)
    fo = {a: i % n_folds for i, a in enumerate(ug)}
    return np.array([fo[a] for a in g])


def inner_split(pool, groups, seed):
    gp = groups[pool]; ug = np.array(sorted(set(gp))); np.random.RandomState(seed).shuffle(ug)
    vg = set(ug[: max(1, int(0.2 * len(ug)))]); isv = np.array([a in vg for a in gp])
    return pool[~isv], pool[isv]


def summarize(res):
    rows = []
    for name, lst in res.items():
        d = pd.DataFrame(lst)
        if len(d) > 1:
            rows.append({"model": name, **{c: f"{d[c].mean():.3f} ± {d[c].std():.3f}" for c in d.columns}})
        else:
            rows.append({"model": name, **{c: f"{d[c].iloc[0]:.3f}" for c in d.columns}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    VERSION = sys.argv[1] if len(sys.argv) > 1 else "with_artist"
    assert VERSION in ("with_artist", "without_artist"), \
        "run: py cnn_covers_keras.py with_artist  (or without_artist)"

    # ---------- data ----------
    m = pd.read_csv("matched_ids.csv", usecols=["id", "msd_track_id", "split"])
    msdi = pd.read_csv("MSD-I_dataset.tsv", sep="\t")[["msd_track_id", "msd_artist_id"]]
    m = m.merge(msdi, on="msd_track_id", how="left")
    assert m["msd_artist_id"].notna().all()
    paths = [f"covers/{t}.jpg" for t in m["msd_track_id"]]
    missing = [p for p in paths if not os.path.exists(p)]
    assert not missing, f"{len(missing)} covers missing: run download_covers.py first"
    imgs = load_images(paths)

    clean = pd.read_csv("spotgentrack_clean.csv")
    clean["split"] = clean["split"].replace({"validation": "val"})
    DROP = ["id", "split", "popularity", "pop_class", "song_id"]     # song_id is never a feature
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

    def tab_prep(tr, others):                              # scaler fitted on the training rows only
        sc = StandardScaler().fit(T.loc[tr, cont_cols])
        def tab(ix):
            X = T.loc[ix].copy(); X[cont_cols] = sc.transform(X[cont_cols])
            return X.values.astype("float32")
        return tab(tr), [tab(o) for o in others]

    CONFIGS = [("mean_baseline", "mean"), ("tabular_mlp", "mlp")]
    if VERSION == "with_artist":
        CONFIGS.append(("cnn_image_only", "cnn"))          # does not depend on the artist features
    CONFIGS.append(("tabular+cnn", "cnn+tab"))

    def run_configs(tr, va, te, seed):
        Xtr, (Xva, Xte) = tab_prep(tr, [va, te])
        out = {}
        for name, kind in CONFIGS:
            if kind == "mean":
                out[name] = np.full(len(te), y[tr].mean())
            elif kind == "mlp":
                out[name] = fit_predict_mlp(Xtr, y[tr], Xva, y[va], Xte, seed)
            elif kind == "cnn":
                out[name] = fit_predict_cnn(imgs[tr], None, y[tr], imgs[va], None, y[va],
                                            imgs[te], None, seed, False)
            else:
                out[name] = fit_predict_cnn(imgs[tr], Xtr, y[tr], imgs[va], Xva, y[va],
                                            imgs[te], Xte, seed, True)
        return out

    # ---------- Evaluation 1: the team's split ----------
    tr, va, te = (np.where(m["split"] == s)[0] for s in ["train", "val", "test"])
    res = {n: [] for n, _ in CONFIGS}
    for s in SEEDS:
        for name, p in run_configs(tr, va, te, s).items():
            res[name].append(metrics(y[te], p))
        print("team split: seed", s, "done", flush=True)
    r1 = summarize(res)
    print("\nTEAM SPLIT (test n=%d), %s" % (len(te), VERSION)); print(r1.to_string(index=False))
    r1.to_csv(f"keras_cnn_results_team_split_{VERSION}.csv", index=False)

    # ---------- Evaluation 2: 5-fold CV, all tracks of an artist stay in one fold ----------
    res_cv = {n: [] for n, _ in CONFIGS}
    for s in SEEDS:
        f = artist_folds(groups, 5, s)
        oof = {n: np.zeros(len(m)) for n, _ in CONFIGS}
        for k in range(5):
            te_k = np.where(f == k)[0]; pool = np.where(f != k)[0]
            tr_k, va_k = inner_split(pool, groups, s)
            for name, p in run_configs(tr_k, va_k, te_k, s).items():
                oof[name][te_k] = p
            print(f"grouped CV: seed {s} fold {k + 1}/5 done", flush=True)
        for name in oof: res_cv[name].append(metrics(y, oof[name]))
    r2 = summarize(res_cv)
    print("\nARTIST-GROUPED 5-FOLD CV (all %d tracks), %s" % (len(m), VERSION)); print(r2.to_string(index=False))
    r2.to_csv(f"keras_cnn_results_grouped_cv_{VERSION}.csv", index=False)
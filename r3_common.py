"""
F20DL Group 5 - shared code for R3 (classical models).

Used by 03_r3_regression.ipynb and 04_r3_classification.ipynb so every model
is trained, tuned and scored in exactly the same way.

Rules this file enforces:
  * Uses the shared split in spotgentrack_clean.csv - never re-splits.
  * Cross-validation is StratifiedGroupKFold (10 folds), grouped by song_id so
    two versions of the same song never sit in a training fold and a validation
    fold at the same time. Folds are stratified by pop_class in BOTH tasks.
  * Scaling happens inside a Pipeline, so the scaler is re-fitted on the
    training part of every fold (no leakage from the held-out fold).
  * The test set is only touched by final_test(), which should be run once,
    at the very end, after the MLP (R4) is finished too.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 10
DATA_FILE = "spotgentrack_clean.csv"
CLASS_ORDER = ["Low", "Medium", "High"]
NON_FEATURES = ["id", "song_id", "split", "popularity", "pop_class"]

# ---------------------------------------------------------------------------
# Data and feature groups
# ---------------------------------------------------------------------------
AUDIO_BASE = ["acousticness", "danceability", "energy", "instrumentalness", "liveness",
              "loudness", "speechiness", "tempo", "valence", "mode"]
ARTIST_BASE = ["artist_popularity", "log_followers"]          # + genre_broad_* one-hots


def load_data(path=DATA_FILE):
    """Return the full cleaned table and the three splits."""
    df = pd.read_csv(path)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return df, train, val, test


def feature_groups(df):
    """
    Feature sets used in the comparison (README section 5, 'Should'):
      all        - every candidate feature
      audio      - the 12 audio descriptors (key and time_signature are one-hot)
      metadata   - everything that is not audio (track, artist, album)
      no_artist  - all features except the artist group
                   (artist_popularity, log_followers, broad genre)
    """
    feats = [c for c in df.columns if c not in NON_FEATURES]
    audio = AUDIO_BASE + [c for c in feats if c.startswith(("key_", "time_signature_"))]
    artist = ARTIST_BASE + [c for c in feats if c.startswith("genre_broad_")]
    metadata = [c for c in feats if c not in audio]
    no_artist = [c for c in feats if c not in artist]
    return {"all": feats, "audio": audio, "metadata": metadata, "no_artist": no_artist}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def grouped_folds(data, n_folds=N_FOLDS, seed=SEED):
    """
    Pre-computed (train_idx, test_idx) pairs for GridSearchCV.
    Stratified by pop_class, grouped by song_id. Pre-computing the folds lets the
    regression task use the same stratified, grouped folds (StratifiedGroupKFold
    cannot stratify on a continuous target directly).
    """
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(data, data["pop_class"], groups=data["song_id"]))


def tuning_subsample(train, n_rows, seed=SEED):
    """
    Stratified random subsample of the training set, used to tune the slow
    models (k-NN, Random Forest). The chosen settings are then re-fitted on the
    FULL training set. CV on the subsample is still grouped by song_id.
    """
    if n_rows is None or n_rows >= len(train):
        return train
    sub, _ = train_test_split(train, train_size=n_rows, stratify=train["pop_class"],
                              random_state=seed)
    return sub.reset_index(drop=True)


def make_pipeline(model, scale=True):
    """Scaler (fitted inside each fold) + model."""
    steps = [("scaler", StandardScaler())] if scale else []
    return Pipeline(steps + [("model", model)])


def tune(name, pipeline, grid, train, features, target, scoring, subsample=None):
    """
    Grid search with grouped stratified 10-fold CV.
    Returns the fitted GridSearchCV object (best model refitted on the data it
    was tuned on) plus a one-row summary.
    """
    data = tuning_subsample(train, subsample)
    folds = grouped_folds(data)
    gs = GridSearchCV(pipeline, {f"model__{k}": v for k, v in grid.items()},
                      scoring=scoring, cv=folds, n_jobs=-1, refit=True)
    t0 = time.time()
    gs.fit(data[features], data[target])
    best = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    i = gs.best_index_
    summary = {"model": name,
               "tuned_on_rows": len(data),
               "best_params": json.dumps(best, default=str),
               "cv_mean": gs.cv_results_["mean_test_score"][i],
               "cv_std": gs.cv_results_["std_test_score"][i],
               "tuning_seconds": round(time.time() - t0, 1)}
    print(f"{name}: best {best} | CV {summary['cv_mean']:.4f} ± {summary['cv_std']:.4f} "
          f"| {len(data):,} rows | {summary['tuning_seconds']}s")
    return gs, summary


def refit_full(gs, train, features, target):
    """Refit the best pipeline from a grid search on the full training set."""
    from sklearn.base import clone
    model = clone(gs.best_estimator_)
    if "model__n_jobs" in model.get_params():      # models were tuned single-threaded
        model.set_params(model__n_jobs=-1)          # because the grid runs in parallel
    model.fit(train[features], train[target])
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}


def classification_metrics(y_true, y_pred):
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "macro_F1": f1_score(y_true, y_pred, average="macro"),
           "weighted_F1": f1_score(y_true, y_pred, average="weighted")}
    rep = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True,
                                zero_division=0)
    for c in CLASS_ORDER:
        out[f"{c}_precision"] = rep[c]["precision"]
        out[f"{c}_recall"] = rep[c]["recall"]
        out[f"{c}_F1"] = rep[c]["f1-score"]
    return out


def high_popularity_error(y_true, y_pred, threshold=60):
    """MAE on the rare highly popular tracks (popularity > threshold) vs the rest."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    hi = y_true > threshold
    return {"MAE_high": mean_absolute_error(y_true[hi], y_pred[hi]),
            "MAE_rest": mean_absolute_error(y_true[~hi], y_pred[~hi]),
            "n_high": int(hi.sum())}


def confusion(y_true, y_pred):
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASS_ORDER),
                        index=[f"true {c}" for c in CLASS_ORDER],
                        columns=[f"pred {c}" for c in CLASS_ORDER])


# ---------------------------------------------------------------------------
# Final test - run ONCE at the end of the project
# ---------------------------------------------------------------------------
def final_test(models, train, test, features, target, task):
    """
    models: dict name -> unfitted pipeline with the chosen settings.
    Fits each on the training set and scores it on the test set.
    Run this once, after all tuning (including the MLP) is finished.
    """
    rows = []
    for name, model in models.items():
        model.fit(train[features], train[target])
        pred = model.predict(test[features])
        m = regression_metrics(test[target], pred) if task == "regression" \
            else classification_metrics(test[target], pred)
        rows.append({"model": name, **m})
    return pd.DataFrame(rows)

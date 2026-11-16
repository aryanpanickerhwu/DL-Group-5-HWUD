"""
R3 (classical models).

- We use the split alredy in spotgentrack_clean.csv and never re-split.
- CV is 10 fold StratifiedGroupKFold. It is grouped by song_id so that two version of the same song doesnt end up in both training and validation fold, and stratified by pop_class in both tasks.
- Scaler is contained within the Pipeline, and therefore, it is fitted using only the training portion of each fold and not the holdout fold.
- The test set will be used in final_test() only. It will run once at the end when MLP (R4) is done.
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
     Feature sets we compare (README section 5):
      all       - every feature
      audio     - the 12 audio features (key and time_signature)
      metadata  - everything that is not audio (track, artist, album)
      no_artist - everything except artist_popularity, log_followers and genre

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
    Create the (train_idx, test_idx) splits upfront, stratified on pop_class, and grouped by song_id. StratifiedGroupKFold cannot stratify on a continuous target, so this approach will allow regression to use the classification folds.

    """
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(data, data["pop_class"], groups=data["song_id"]))


def tuning_subsample(train, n_rows, seed=SEED):
    """
    A smaller stratified sample of train data is used for tuning slow models (k-NN, RF).
    The optimal values are refitted on the whole training dataset.
    Returns the whole dataset if n_rows is None.

    """
    if n_rows is None or n_rows >= len(train):
        return train
    sub, _ = train_test_split(train, train_size=n_rows, stratify=train["pop_class"],
                              random_state=seed)
    return sub.reset_index(drop=True)


def make_pipeline(model, scale=True):

    """Scaler + model. Set scale=false for models that doesnt need it (eg. trees)"""
    steps = [("scaler", StandardScaler())] if scale else []
    return Pipeline(steps + [("model", model)])


def tune(name, pipeline, grid, train, features, target, scoring, subsample=None):
    """
    Grid search using the grouped 10-fold CV above.
    it returns the fitted GridSearchCV and a summary dict (best params, cv mean, cv std, time).

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

    """Take the best pipeline from tune() and fit it on the full training set."""
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
    """ MSE, RMSE, MSE AND R2 """
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

    """
    MAE on the rare very popular tracks (popularity > threshold) compared to
    the rest, to check if the model struggles with hits.
    Also returns n_high.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    hi = y_true > threshold
    return {"MAE_high": mean_absolute_error(y_true[hi], y_pred[hi]),
            "MAE_rest": mean_absolute_error(y_true[~hi], y_pred[~hi]),
            "n_high": int(hi.sum())}


def confusion(y_true, y_pred):

    """ Confusion matrix as a dataframe, rows are true classes and cloumns are the predictions"""
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASS_ORDER),
                        index=[f"true {c}" for c in CLASS_ORDER],
                        columns=[f"pred {c}" for c in CLASS_ORDER])


# ---------------------------------------------------------------------------
# Final test - run ONCE at the end of the project
# ---------------------------------------------------------------------------
def final_test(models, train, test, features, target, task):

    """
    Train the model using the training set and then evaluate it using the testing set.
    The models variable is a dict of {name: Pipeline object using the optimal settings},
    Task is "regression" or "classification". This is only done after tuning (MLP included).
    """
    rows = []
    for name, model in models.items():
        model.fit(train[features], train[target])
        pred = model.predict(test[features])
        m = regression_metrics(test[target], pred) if task == "regression" \
            else classification_metrics(test[target], pred)
        rows.append({"model": name, **m})
    return pd.DataFrame(rows)

"""
F20DL Group 5 - full data preparation for SpotGenTrack.
Replaces make_joined_and_split.py (its split leaked duplicate songs across splits).

Run from the folder containing spotify_tracks.csv, spotify_artists.csv, spotify_albums.csv.
Outputs:
  spotgentrack_clean.csv : cleaned + encoded features, popularity, pop_class, split
  split_ids.csv          : id, popularity, pop_class, split   (shared split - do not re-split)

Scaling is NOT applied here: fit any scaler on the train rows only, inside your own notebook.
"""
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
log = []                                   # decisions log for the R2 write-up
def note(msg): log.append(msg); print("-", msg)

# ---------------- 1. Load and join -------------------------------------------------
track_cols = ["id", "name", "album_id", "artists_id", "popularity",
              "acousticness", "danceability", "energy", "instrumentalness", "liveness",
              "loudness", "speechiness", "tempo", "valence", "key", "mode", "time_signature",
              "duration_ms", "disc_number", "track_number"]
tracks = pd.read_csv("spotify_tracks.csv", usecols=track_cols)
artists = (pd.read_csv("spotify_artists.csv", usecols=["id", "artist_popularity", "followers", "genres"])
             .drop_duplicates("id").rename(columns={"id": "artist_id"}))
albums = (pd.read_csv("spotify_albums.csv", usecols=["id", "album_type", "release_date", "total_tracks"])
            .drop_duplicates("id").rename(columns={"id": "album_id"}))
tracks["artist_id"] = tracks["artists_id"].apply(lambda x: ast.literal_eval(x)[0])   # main artist
df = (tracks.drop(columns="artists_id")
            .merge(artists, on="artist_id", how="left")
            .merge(albums, on="album_id", how="left"))
note(f"Joined tracks + artists (first listed artist) + albums: {len(df):,} tracks, "
     f"{df.isna().any(axis=1).sum()} rows with missing values")

# ---------------- 2. Cleaning ---------------------------------------------------------
bad = (df["tempo"] == 0) | (df["time_signature"] == 0)
df = df[~bad].copy()
note(f"Dropped {bad.sum()} tracks with tempo = 0 or time_signature = 0 (failed audio analysis)")
note("Kept short (<30 s) and long (>15 min) tracks, loudness > 0 dB, very large track/disc numbers "
     "and pre-1950 release years: these are real, if unusual, values")

# ---------------- 3. Feature transformations -----------------------------------------
df["log_followers"] = np.log1p(df["followers"])
df["log_duration"] = np.log1p(df["duration_ms"])
note(f"log(1+x) on followers (skew {df['followers'].skew():.1f} -> {df['log_followers'].skew():.2f}) "
     f"and duration_ms (skew {df['duration_ms'].skew():.1f} -> {df['log_duration'].skew():.2f})")
df["release_year"] = pd.to_numeric(df["release_date"].astype(str).str[:4], errors="coerce")
note("release_date converted to release_year (dates have mixed day/month/year precision)")

# Broad genre from the artist's genre list (first matching rule wins)
RULES = [("latin", ["latin", "reggaeton", "salsa", "bachata", "cumbia", "mexican", "espanol", "brazil", "sertanejo", "funk carioca"]),
         ("hiphop_rap", ["hip hop", "rap", "trap", "drill", "grime"]),
         ("metal", ["metal", "core"]),
         ("rock", ["rock", "punk", "grunge"]),
         ("rnb_soul", ["r&b", "soul", "funk", "motown"]),
         ("country", ["country"]),
         ("jazz_blues", ["jazz", "blues"]),
         ("classical", ["classical", "orchestra", "baroque", "opera", "romantic era"]),
         ("folk_indie", ["folk", "indie", "singer-songwriter", "acoustic"]),
         ("electronic", ["edm", "house", "techno", "trance", "electro", "dubstep", "drum and bass", "tropical"]),
         ("pop", ["pop"])]
def broad(genres):
    gl = ast.literal_eval(genres) if isinstance(genres, str) else []
    if not gl: return "unknown"
    for label, keys in RULES:
        if any(k in g for g in gl for k in keys): return label
    return "other"
df["genre_broad"] = df["genres"].apply(broad)
note("Grouped 2,371 artist genre labels into 11 broad genres + 'other' + 'unknown' (keyword rules). "
     "Genre shares (%): " + ", ".join(f"{k} {v:.1f}" for k, v in (df['genre_broad'].value_counts(normalize=True) * 100).items()))

# One-hot encoding (categorical features stored as numbers are really categories)
cat_cols = ["key", "time_signature", "album_type", "genre_broad"]
df[["key", "time_signature"]] = df[["key", "time_signature"]].astype(int)
df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)
df["mode"] = df["mode"].astype(int)
note("One-hot encoded key, time_signature, album_type and genre_broad; mode kept as 0/1 (major/minor)")

# ---------------- 4. Target classes ---------------------------------------------------
df["pop_class"] = pd.cut(df["popularity"], bins=[-1, 40, 60, 100], labels=["Low", "Medium", "High"]).astype(str)
note("pop_class by custom binning: Low 0-40, Medium 41-60, High 61-100")

# ---------------- 5. Group-aware stratified 70/15/15 split ----------------------------
# Versions of the same song (same main artist + title, e.g. single and album versions)
# must stay in the same split, otherwise near-identical tracks leak from train into test.
df["song_group"] = df["artist_id"] + "|" + df["name"].astype(str).str.lower().str.strip()
groups = df.groupby("song_group")["pop_class"].agg(lambda s: s.mode().iloc[0]).reset_index()
g_train, g_rest = train_test_split(groups, test_size=0.30, stratify=groups["pop_class"], random_state=SEED)
g_val, g_test = train_test_split(g_rest, test_size=0.50, stratify=g_rest["pop_class"], random_state=SEED)
split_map = {**dict.fromkeys(g_train["song_group"], "train"),
             **dict.fromkeys(g_val["song_group"], "val"),
             **dict.fromkeys(g_test["song_group"], "test")}
df["split"] = df["song_group"].map(split_map)
n_dup = df["song_group"].duplicated(keep=False).sum()
note(f"Split 70/15/15 at song level (artist + title), stratified by pop_class, seed {SEED}; "
     f"{n_dup:,} tracks share a song with another version and are kept together")

# ---------------- 6. Checks ------------------------------------------------------------
ids = {s: set(df.loc[df["split"] == s, "id"]) for s in ["train", "val", "test"]}
assert df["split"].notna().all()
assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]), "ID overlap"
assert df.groupby("song_group")["split"].nunique().max() == 1, "song spans splits"
counts = df["split"].value_counts().reindex(["train", "val", "test"])
note("Split sizes: " + ", ".join(f"{k} {v:,} ({v/len(df)*100:.1f}%)" for k, v in counts.items()))
mix = (pd.crosstab(df["split"], df["pop_class"], normalize="index") * 100).round(2) \
        .reindex(index=["train", "val", "test"], columns=["Low", "Medium", "High"])
print(mix.to_string())
note("Checks passed: no ID overlap, no song in more than one split, every track assigned")

# ---------------- 7. Save ------------------------------------------------------------------
# song_id: integer id per song (same main artist + title). Use it as `groups` in StratifiedGroupKFold
# so versions of the same song never end up in both a training fold and a validation fold.
df["song_id"] = pd.factorize(df["song_group"])[0]
drop = ["name", "album_id", "artist_id", "genres", "release_date", "followers", "duration_ms", "song_group"]
out = df.drop(columns=drop)
first = ["id", "song_id", "split", "popularity", "pop_class"]
out = out[first + [c for c in out.columns if c not in first]]
out.to_csv("spotgentrack_clean.csv", index=False)
out[["id", "song_id", "popularity", "pop_class", "split"]].to_csv("split_ids.csv", index=False)
with open("data_preparation_log_raw.txt", "w") as f:
    f.write("\n".join(log) + "\n\nClass mix per split (%):\n" + mix.to_string() + "\n")
print(f"\nSaved spotgentrack_clean.csv ({out.shape[0]:,} rows x {out.shape[1]} cols), split_ids.csv, data_preparation_log_raw.txt")

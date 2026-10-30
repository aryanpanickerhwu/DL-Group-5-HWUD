"""
F20DL Group 5 - build the joined SpotGenTrack table, popularity classes and the shared split.

Outputs (same folder as this script):
  spotgentrack_joined.csv : one row per track, 21 candidate features + popularity + pop_class
  split_ids.csv           : id, popularity, pop_class, split (train / val / test)

Method:
  - ID column: Spotify track `id` (unique, 101,939 tracks)
  - Classes (custom binning): Low 0-40, Medium 41-60, High 61-100
  - Split: 70 / 15 / 15, stratified by pop_class, random_state = SEED
    (70% train first, then the remaining 30% split 50/50 into val and test)
"""
import ast
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# ---- 1. Load ---------------------------------------------------------------
track_cols = ["id", "album_id", "artists_id", "popularity",
              "acousticness", "danceability", "energy", "instrumentalness", "liveness",
              "loudness", "speechiness", "tempo", "valence", "key", "mode", "time_signature",
              "duration_ms", "disc_number", "track_number"]
tracks = pd.read_csv("spotify_tracks.csv", usecols=track_cols)
artists = (pd.read_csv("spotify_artists.csv", usecols=["id", "artist_popularity", "followers", "genres"])
             .drop_duplicates("id").rename(columns={"id": "artist_id"}))
albums = (pd.read_csv("spotify_albums.csv", usecols=["id", "album_type", "release_date", "total_tracks"])
            .drop_duplicates("id").rename(columns={"id": "album_id"}))

# ---- 2. Join (first listed artist = main artist) ---------------------------
tracks["artist_id"] = tracks["artists_id"].apply(lambda x: ast.literal_eval(x)[0])
df = (tracks.drop(columns="artists_id")
            .merge(artists, on="artist_id", how="left")
            .merge(albums, on="album_id", how="left"))
assert len(df) == len(tracks) and df["id"].is_unique

# ---- 3. Popularity classes --------------------------------------------------
df["pop_class"] = pd.cut(df["popularity"], bins=[-1, 40, 60, 100],
                         labels=["Low", "Medium", "High"]).astype(str)

# ---- 4. Stratified 70/15/15 split -------------------------------------------
train_ids, rest_ids = train_test_split(df["id"], test_size=0.30,
                                       stratify=df["pop_class"], random_state=SEED)
rest = df.set_index("id").loc[rest_ids]
val_ids, test_ids = train_test_split(rest.index.to_series(), test_size=0.50,
                                     stratify=rest["pop_class"], random_state=SEED)
df["split"] = "train"
df.loc[df["id"].isin(val_ids), "split"] = "val"
df.loc[df["id"].isin(test_ids), "split"] = "test"

# ---- 5. Checks ----------------------------------------------------------------
s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
assert not (s_train & s_val) and not (s_train & s_test) and not (s_val & s_test), "ID overlap!"
assert len(s_train) + len(s_val) + len(s_test) == len(df), "rows missing"

print("Rows:", len(df), "| seed:", SEED)
print("\nSplit sizes:")
print(df["split"].value_counts().reindex(["train", "val", "test"]).to_string())
print("\nClass % per split:")
print((pd.crosstab(df["split"], df["pop_class"], normalize="index") * 100).round(2)
        .reindex(index=["train", "val", "test"], columns=["Low", "Medium", "High"]).to_string())
print("\nMissing values after join (%):")
m = (df.isna().mean() * 100).round(3); print(m[m > 0].to_string() if (m > 0).any() else "none")

# ---- 6. Save ------------------------------------------------------------------
df.to_csv("spotgentrack_joined.csv", index=False)
df[["id", "popularity", "pop_class", "split"]].to_csv("split_ids.csv", index=False)
print("\nSaved spotgentrack_joined.csv and split_ids.csv")

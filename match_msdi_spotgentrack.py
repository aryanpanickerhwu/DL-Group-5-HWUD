"""
Match MSD-I tracks to SpotGenTrack tracks.
There is no shared ID, so we match on (artist name + song title) after cleaning both.
Output: msdi_spotgentrack_matches.csv
"""
import ast
import re
import pandas as pd

# --- 1. Cleaning function: make names comparable --------------------------------
def norm(s):
    s = str(s).lower()                           # "Blood" == "blood"
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)        # drop "(Remastered)", "[Live]"
    s = re.sub(r"\s-\s.*$", "", s)               # drop " - Remastered 2011"
    s = re.sub(r"\b(feat|ft)\.?\b.*$", "", s)    # drop "feat. X"
    s = re.sub(r"[^a-z0-9 ]", "", s)             # drop punctuation
    return re.sub(r"\s+", " ", s).strip()        # tidy spaces

# --- 2. MSD-I only has IDs, so add artist + title from MSD's unique_tracks.txt ---
msdi = pd.read_csv("MSD-I_dataset.tsv", sep="\t")
names = pd.read_csv("msdi-data/unique_tracks.txt", sep="<SEP>", engine="python",
                    names=["msd_track_id", "song_id", "artist", "title"], quoting=3)
msdi = msdi.merge(names, on="msd_track_id", how="left")

# --- 3. SpotGenTrack: tracks only store artist IDs, so look up the artist name ---
tracks = pd.read_csv("spotify_tracks.csv", usecols=["id", "name", "artists_id", "popularity"])
artists = (pd.read_csv("spotify_artists.csv", usecols=["id", "name"])
             .drop_duplicates("id")
             .rename(columns={"id": "artist_id", "name": "artist"}))
tracks["artist_id"] = tracks["artists_id"].apply(lambda x: ast.literal_eval(x)[0])  # first artist
sp = tracks.merge(artists, on="artist_id", how="left")

# --- 4. Build a match key "artist|title" on both sides and compare ---------------
sp["key"] = sp["artist"].map(norm) + "|" + sp["name"].map(norm)
msdi["key"] = msdi["artist"].map(norm) + "|" + msdi["title"].map(norm)

matched = msdi[msdi["key"].isin(set(sp["key"]))]
print("MSD-I tracks also in SpotGenTrack:", matched["msd_track_id"].nunique())   # 442

# --- 5. Save matches with their Spotify popularity -------------------------------
out = matched.merge(sp[["key", "popularity"]].drop_duplicates("key"), on="key")
out[["msd_track_id", "artist", "title", "genre", "set", "popularity"]] \
    .to_csv("msdi_spotgentrack_matches.csv", index=False)

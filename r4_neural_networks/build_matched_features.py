import ast, re, urllib.request
import numpy as np, pandas as pd

def norm(s):                                   # identical to the team's script
    s = str(s).lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    s = re.sub(r"\s-\s.*$", "", s)
    s = re.sub(r"\b(feat|ft)\.?\b.*$", "", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()

# ---- 1. MSD-I tracks the team matched (artist/title are in this file) ----
m = pd.read_csv("msdi_spotgentrack_matches.csv")
print("rows in matches file:", len(m), "| unique msd_track_id:", m["msd_track_id"].nunique())
m["key"] = m["artist"].map(norm) + "|" + m["title"].map(norm)

# ---- 2. SpotGenTrack side: rebuild keys but KEEP the track id ----
tracks = pd.read_csv("spotify_tracks.csv", usecols=["id", "name", "artists_id"])
artists = (pd.read_csv("spotify_artists.csv", usecols=["id", "name"])
             .drop_duplicates("id").rename(columns={"id": "artist_id", "name": "artist"}))
tracks["artist_id"] = tracks["artists_id"].apply(lambda x: ast.literal_eval(x)[0])
sp = tracks.merge(artists, on="artist_id", how="left")
sp["key"] = sp["artist"].map(norm) + "|" + sp["name"].map(norm)

pairs = (m[["msd_track_id", "key"]].drop_duplicates()
         .merge(sp[["key", "id"]].drop_duplicates(), on="key"))
print("MSD-I tracks with a key match:", pairs["msd_track_id"].nunique())

# ---- 3. Keep ids that survived cleaning; attach the team's split + popularity ----
clean = pd.read_csv("spotgentrack_clean.csv", usecols=["id", "split", "popularity"])
pairs = pairs.merge(clean, on="id", how="inner")
print("MSD-I tracks matched to the clean data:", pairs["msd_track_id"].nunique())

n_ids = pairs.groupby("msd_track_id")["id"].nunique()
print("clean ids per MSD-I track:\n", n_ids.value_counts())
final = (pairs[pairs["msd_track_id"].isin(n_ids[n_ids == 1].index)]
         .drop(columns="key").drop_duplicates().reset_index(drop=True))
final["split"] = final["split"].replace({"validation": "val"})

# ---- 4. Attach the precomputed cover features ----
base = "msdi-data/"
ids, vis, mm = [], [], []
for s in ["train", "val", "test"]:
    idx = pd.read_csv(f"{base}splits/items_index_{s}_MSD-I.tsv", sep="\t", header=None)[0]
    xv = np.load(f"{base}train_data/X_{s}_visual_MSD-I.npy")
    xm = np.load(f"{base}train_data/X_{s}_mm-visual_MSD-I.npy")
    assert len(idx) == len(xv) == len(xm), f"{s}: index/feature size mismatch"
    ids.append(idx); vis.append(xv); mm.append(xm)
all_ids = pd.concat(ids, ignore_index=True)
all_vis, all_mm = np.vstack(vis), np.vstack(mm)
assert all_ids.is_unique
pos = pd.Series(np.arange(len(all_ids)), index=all_ids.values)

rows = pos.reindex(final["msd_track_id"])
print("matched tracks with no cover features:", int(rows.isna().sum()))
final = final[rows.notna().values].reset_index(drop=True)
rows = rows.dropna().astype(int).values

print("\nFINAL matched set:", len(final))
print(final["split"].value_counts())
final.to_csv("matched_ids.csv", index=False)
np.savez_compressed("matched_visual_features.npz",
                    id=final["id"].astype(str).values,
                    msd_track_id=final["msd_track_id"].astype(str).values,
                    split=final["split"].astype(str).values,
                    popularity=final["popularity"].values,
                    X_visual=all_vis[rows], X_mm=all_mm[rows])
print("saved matched_ids.csv and matched_visual_features.npz")

# ---- 5. Optional: are the cover image links still alive? (20 tries) ----
msdi = pd.read_csv("MSD-I_dataset.tsv", sep="\t").set_index("msd_track_id")
ok = 0
test_ids = final["msd_track_id"].head(20)
for t in test_ids:
    try:
        req = urllib.request.Request(msdi.loc[t, "image_url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            ok += int(r.status == 200)
    except Exception:
        pass
print(f"image links working: {ok}/{len(test_ids)}")
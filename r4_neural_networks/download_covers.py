import os, time, urllib.request
import pandas as pd

m = pd.read_csv("matched_ids.csv")
msdi = pd.read_csv("MSD-I_dataset.tsv", sep="\t").set_index("msd_track_id")
os.makedirs("covers", exist_ok=True)

failed = []
for t in m["msd_track_id"]:
    out = f"covers/{t}.jpg"
    if os.path.exists(out):
        continue
    try:
        req = urllib.request.Request(msdi.loc[t, "image_url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r, open(out, "wb") as f:
            f.write(r.read())
        time.sleep(0.2)
    except Exception as e:
        failed.append((t, str(e)))

print("downloaded:", len(m) - len(failed), "| failed:", len(failed))
for t, e in failed[:10]:
    print(t, e)
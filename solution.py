"""Team Nova - Phase 2 submission (Round 1).

Model: SmoothedCTR, the best model on our validation (hit@1 0.525 vs 0.441 for
the organisers' most-clicked example). For every item we compute a recency-weighted,
smoothed click rate from the feedback in data_dir (computed at predict time, no
pre-trained weights). For every request we recommend the highest-scoring item that
is available at the request's timestamp: valid_from <= t < valid_to.
New users get the same best available item.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 1.0      # tuned on validation (round 1)
PRIOR_STRENGTH = 100.0    # tuned on validation (round 1)
TOP_K = 512               # candidates checked per row before falling back to a full scan
DAY = 86400


def load_feedback(data):
    """Long format: one row per displayed item, clicked 0/1."""
    pq = data / "feedback_long.parquet"
    csv = data / "feedback_long.csv"
    try:
        if pq.exists():
            return pd.read_parquet(pq, columns=["timestamp", "itemID", "clicked"])
    except Exception:
        pass
    if csv.exists():
        return pd.read_csv(csv, usecols=["timestamp", "itemID", "clicked"], dtype={"itemID": str})
    rows = []
    with open(data / "feedback.jsonl") as f:
        for line in f:
            e = json.loads(line)
            for it in e["displayed_items"]:
                rows.append((e["timestamp"], it, int(it == e.get("clicked_item"))))
    return pd.DataFrame(rows, columns=["timestamp", "itemID", "clicked"])


def item_scores(fb, items):
    fb = fb.astype({"itemID": str})
    ts = fb["timestamp"].astype("int64").to_numpy()
    w = 0.5 ** ((ts.max() - ts) / (HALF_LIFE_DAYS * DAY))
    g = pd.DataFrame({"itemID": fb["itemID"].to_numpy(), "w": w,
                      "wc": w * fb["clicked"].astype(float).to_numpy()}).groupby("itemID")[["w", "wc"]].sum()
    global_ctr = g["wc"].sum() / max(g["w"].sum(), 1e-9)
    ctr = (g["wc"] + PRIOR_STRENGTH * global_ctr) / (g["w"] + PRIOR_STRENGTH)
    return ctr.reindex(items).fillna(global_ctr).to_numpy()


def predict(data_dir, batch_size=256):
    data = Path(data_dir)
    rows = pd.read_csv(data / "x_pred.csv", dtype={"rowid": str, "user_id": str})
    rows = rows.drop_duplicates("rowid")
    cat = pd.read_csv(data / "catalog.csv", dtype={"itemID": str}).drop_duplicates("itemID")

    items = cat["itemID"].to_numpy()
    vf = cat["valid_from"].to_numpy(dtype="int64")
    vt = cat["valid_to"].to_numpy(dtype="int64")

    try:
        score = item_scores(load_feedback(data), items)
    except Exception as e:
        print("scoring failed, using catalogue order:", e)
        score = np.zeros(len(items))

    order = np.argsort(-score, kind="stable")
    items_s, vf_s, vt_s = items[order], vf[order], vt[order]
    k = min(TOP_K, len(items_s))

    rowids = rows["rowid"].to_numpy()
    ts = rows["timestamp"].to_numpy(dtype="int64")
    for start in range(0, len(rowids), batch_size):
        r = rowids[start:start + batch_size]
        t = ts[start:start + batch_size][:, None]
        ok = (vf_s[None, :k] <= t) & (t < vt_s[None, :k])
        has = ok.any(1)
        chosen = items_s[np.where(has, ok.argmax(1), 0)].copy()
        for j in np.where(~has)[0]:
            full = np.where((vf_s <= t[j, 0]) & (t[j, 0] < vt_s))[0]
            chosen[j] = items_s[full[0]] if len(full) else items_s[0]
        yield [(str(a), str(b)) for a, b in zip(r, chosen)]
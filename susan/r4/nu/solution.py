"""Team Nova R3 candidate - Nova User: k-adjusted global appeal + small co-click taste term.
pick(row) = argmax over items LIVE at the row's timestamp of
            log appeal_OE(i) + W_KNN * z(co-click similarity of i to the user's history clicks) [+ W_RECLICK * clicked_before]
Everything is computed from data_dir (no bundled ids). numpy/pandas only, no model fits beyond counting.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorer as nu  # vendored copy of r3/scorers/nova_user/scorer.py


def _read_feedback(d):
    pq = d / "feedback_long.parquet"
    if pq.exists():
        try:
            import pyarrow.parquet as pqm
            return pqm.read_table(pq, columns=["event_id", "user_id", "timestamp", "itemID", "clicked"],
                                  read_dictionary=["event_id", "user_id", "itemID"]).to_pandas()
        except ImportError:
            return pd.read_parquet(pq)
    if (d / "feedback_long.csv").exists():
        return pd.read_csv(d / "feedback_long.csv", dtype={"event_id": "category", "user_id": "category", "itemID": "category"})
    rows = []
    import json
    with open(d / "feedback.jsonl") as f:
        for line in f:
            e = json.loads(line)
            for it in e["displayed_items"]:
                rows.append((e["event_id"], e["user_id"], int(e["timestamp"]), it, int(it == e.get("clicked_item"))))
    return pd.DataFrame(rows, columns=["event_id", "user_id", "timestamp", "itemID", "clicked"])


def _read(d, name):
    p = d / (name + ".parquet")
    return pd.read_parquet(p) if p.exists() else pd.read_csv(d / (name + ".csv"))


def predict(data_dir, batch_size=256):
    d = Path(data_dir)
    fb = _read_feedback(d)
    cat = _read(d, "catalog").astype({"itemID": str}).drop_duplicates("itemID")
    rows = _read(d, "x_pred").astype({"rowid": str, "user_id": str})
    rows["timestamp"] = rows["timestamp"].astype("int64")
    rows = rows.drop_duplicates("rowid").reset_index(drop=True)

    sc = nu.Scorer().fit(fb, cat)
    del fb
    # candidate items = catalogue items only (never recommend items outside it)
    ci = sc.items.get_indexer(cat.itemID)
    cand_items = cat.itemID.to_numpy()
    vf = cat.valid_from.to_numpy("int64"); vt = cat.valid_to.to_numpy("int64")
    ap = np.where(ci >= 0, sc.ap_all[np.maximum(ci, 0)], sc.ap_cold)
    NI = len(sc.items)

    ui = sc.users.get_indexer(rows.user_id)
    order = np.argsort(ui, kind="stable")
    pick = np.empty(len(rows), dtype=object)
    uq, st = np.unique(ui[order], return_index=True)
    en = np.r_[st[1:], len(order)]
    CH = 64
    for a in range(0, len(uq), CH):
        us = uq[a:a + CH]
        lo, hi = st[a], en[min(a + CH, len(uq)) - 1]
        ridx = order[lo:hi]
        loc = np.searchsorted(us, ui[ridx])
        S = np.tile(ap.astype(np.float32), (len(us), 1))
        known = us >= 0
        if known.any() and (sc.w_knn or sc.w_reclick):
            add = np.zeros((int(known.sum()), len(ci)), np.float32)
            if sc.w_knn:
                M = sc._knn_dense(us[known])[:, np.maximum(ci, 0)] / sc.knn_sd
                M[:, ci < 0] = 0.0
                add += sc.w_knn * M
            if sc.w_reclick:
                for r, u in enumerate(us[known]):
                    js = sc.u_items[sc.u_ptr[u]:sc.u_ptr[u + 1]]
                    hit = np.isin(ci, js)
                    add[r, hit] += sc.w_reclick
            S[known] += add
        # all unseen users share code -1, so one "user" can own a huge number of rows:
        # do the rows x items matrix in row chunks so memory stays flat
        RCH = 4096
        for b0 in range(0, len(ridx), RCH):
            rr = ridx[b0:b0 + RCH]
            R = S[loc[b0:b0 + RCH]]                       # rows x catalogue items
            t = rows.timestamp.to_numpy()[rr][:, None]
            R[(vf[None, :] > t) | (t >= vt[None, :])] = -np.inf   # liveness on every row
            pick[rr] = cand_items[R.argmax(1)]
            bad = ~np.isfinite(R.max(1))
            if bad.any():
                raise RuntimeError(f"{int(bad.sum())} rows have no live item")
            del R
    rid = rows.rowid.to_numpy()
    for s0 in range(0, len(rid), batch_size):
        yield [(str(r), str(p)) for r, p in zip(rid[s0:s0 + batch_size], pick[s0:s0 + batch_size])]

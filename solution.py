"""Team Nova - Phase 2 submission.

Model: slate choice model (MNL) blended with a smoothed click rate.
- MNL: P(click item i | slate) = exp(b_i) / (exp(c) + sum_j exp(b_j)), where c is a
  "no click" score per time bin + per user. Fit on the feedback in data_dir with a
  1 day half-life on old events. Takes ~1s.
- score = 0.7 * b_i + 0.3 * log(smoothed ctr)   (holdout: 0.531 vs 0.523 plain ctr)
- if the MNL fails we fall back to the plain smoothed ctr (half life 1, prior 100).
For each request we pick the best scoring item with valid_from <= t < valid_to.
- per user: if the user clicked a live item in the last 12h we pick that one instead
  (re-click rate 0-12h after a click was 21/41 = 0.51 in the feedback, top items run ~0.15-0.28).
  after 12h the re-click rate drops to the same level as the top items so we leave it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 1.0   # recency decay for both models
PRIOR_STRENGTH = 100.0  # smoothing of the click rate
LAM = 5.0              # L2 on item scores
LAM_U = 0.3            # L2 on user no-click scores
TIME_BINS = 24         # no-click score bins over the feedback period
W_MNL = 0.7            # blend weight
TOP_K = 512            # candidates checked per row before a full scan
DAY = 86400
RECLICK_HOURS = 12.0   # only use the user's own clicks this recent


def read_any(data, name, **kw):
    # try parquet first, then csv
    pq, csv = data / (name + ".parquet"), data / (name + ".csv")
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            print("parquet read failed for", name, e)
    return pd.read_csv(csv, **kw)


def load_feedback(data):
    # long format: event_id, user_id, timestamp, itemID, clicked
    try:
        return read_any(data, "feedback_long", dtype={"itemID": str, "user_id": str, "event_id": str})
    except Exception as e:
        print("no feedback_long, reading jsonl:", e)
    rows = []
    with open(data / "feedback.jsonl") as f:
        for line in f:
            e = json.loads(line)
            for it in e["displayed_items"]:
                rows.append((e["event_id"], e["user_id"], e["timestamp"], it, int(it == e.get("clicked_item"))))
    return pd.DataFrame(rows, columns=["event_id", "user_id", "timestamp", "itemID", "clicked"])


def smoothed_ctr(fb, items):
    ts = fb["timestamp"].astype("int64").to_numpy()
    w = 0.5 ** ((ts.max() - ts) / (HALF_LIFE_DAYS * DAY))
    g = pd.DataFrame({"itemID": fb["itemID"].to_numpy(), "w": w,
                      "wc": w * fb["clicked"].astype(float).to_numpy()}).groupby("itemID")[["w", "wc"]].sum()
    g_ctr = fb["clicked"].astype(float).mean()
    ctr = (g["wc"] + PRIOR_STRENGTH * g_ctr) / (g["w"] + PRIOR_STRENGTH)
    return ctr.reindex(items).fillna(g_ctr).to_numpy()


def fit_mnl(fb):
    from scipy.optimize import minimize
    fb = fb.sort_values("event_id").reset_index(drop=True)
    ev, _ = pd.factorize(fb["event_id"])
    it, item_ids = pd.factorize(fb["itemID"])
    E, I = ev.max() + 1, len(item_ids)
    clk = fb["clicked"].astype(float).to_numpy()
    first = fb.groupby(ev).first()
    et = first["timestamp"].to_numpy(dtype=float)
    eu, users = pd.factorize(first["user_id"])
    U = len(users)
    eclk = np.bincount(ev, weights=clk, minlength=E)  # 1 if the event had a click
    we = 0.5 ** ((et.max() - et) / (HALF_LIFE_DAYS * DAY))
    tb = np.minimum(((et - et.min()) / (et.max() - et.min() + 1) * TIME_BINS).astype(int), TIME_BINS - 1)

    def loss(x):
        b, c, cu = x[:I], x[I:I + TIME_BINS], x[I + TIME_BINS:]
        eb = np.exp(b[it])
        ce = c[tb] + cu[eu]
        eo = np.exp(ce)
        den = np.bincount(ev, weights=eb, minlength=E) + eo
        ll = np.bincount(ev, weights=clk * b[it], minlength=E) + (1 - eclk) * ce - np.log(den)
        f = -(we * ll).sum() + 0.5 * LAM * (b @ b) + 0.5 * LAM_U * (cu @ cu)
        gb = -np.bincount(it, weights=we[ev] * (clk - eb / den[ev]), minlength=I) + LAM * b
        r = we * ((1 - eclk) - eo / den)
        gc = -np.bincount(tb, weights=r, minlength=TIME_BINS)
        gu = -np.bincount(eu, weights=r, minlength=U) + LAM_U * cu
        return f, np.concatenate([gb, gc, gu])

    x0 = np.zeros(I + TIME_BINS + U)
    x0[I:I + TIME_BINS] = 2.0
    res = minimize(loss, x0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return pd.Series(res.x[:I], index=item_ids)


def item_scores(fb, items):
    fb = fb.astype({"itemID": str, "user_id": str, "event_id": str})
    ctr = smoothed_ctr(fb, items)
    try:
        b = fit_mnl(fb).reindex(items).fillna(0.0).to_numpy()
        s = W_MNL * b + (1 - W_MNL) * np.log(ctr)
        if not np.isfinite(s).all():
            raise ValueError("non finite mnl scores")
        return s
    except Exception as e:
        print("mnl failed, using plain ctr:", e)
        return ctr


def user_picks(fb, rows, vf_map, vt_map):
    # rowid -> item the user clicked in the last RECLICK_HOURS that is live at t (latest click wins)
    try:
        clk = fb[fb["clicked"].astype(int) == 1][["user_id", "itemID", "timestamp"]].astype({"user_id": str, "itemID": str})
        clk = clk.rename(columns={"timestamp": "tc"})
        m = rows[["rowid", "user_id", "timestamp"]].merge(clk, on="user_id")
        gap = m["timestamp"].astype("int64") - m["tc"].astype("int64")
        m = m[(gap >= 0) & (gap <= RECLICK_HOURS * 3600)].copy()
        m["vf"] = m["itemID"].map(vf_map)
        m["vt"] = m["itemID"].map(vt_map)
        t = m["timestamp"].astype("int64")
        m = m[m["vf"].notna() & (m["vf"] <= t) & (t < m["vt"])]
        m = m.sort_values("tc").drop_duplicates("rowid", keep="last")
        return dict(zip(m["rowid"], m["itemID"]))
    except Exception as e:
        print("user picks failed, global only:", e)
        return {}


def predict(data_dir, batch_size=256):
    data = Path(data_dir)
    rows = read_any(data, "x_pred", dtype={"rowid": str, "user_id": str})
    rows = rows.astype({"rowid": str}).drop_duplicates("rowid")
    cat = read_any(data, "catalog", dtype={"itemID": str}).astype({"itemID": str}).drop_duplicates("itemID")

    items = cat["itemID"].to_numpy()
    vf = cat["valid_from"].to_numpy(dtype="int64")
    vt = cat["valid_to"].to_numpy(dtype="int64")

    fb = None
    try:
        fb = load_feedback(data)
        score = item_scores(fb, items)
    except Exception as e:
        print("scoring failed, using catalogue order:", e)
        score = np.zeros(len(items))
    upick = {}
    if fb is not None:
        upick = user_picks(fb, rows, pd.Series(vf, index=items), pd.Series(vt, index=items))
    print("user picks:", len(upick))

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
            # nothing live in the top k, scan the whole catalogue
            full = np.where((vf_s <= t[j, 0]) & (t[j, 0] < vt_s))[0]
            chosen[j] = items_s[full[0]] if len(full) else items_s[0]
        out = []
        for a, b in zip(r, chosen):
            out.append((str(a), str(upick.get(a, b))))
        yield out

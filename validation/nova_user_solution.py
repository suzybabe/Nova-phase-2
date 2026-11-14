import json
from pathlib import Path

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 1.0
PRIOR_STRENGTH = 100.0
LAM = 5.0
LAM_U = 0.3
TIME_BINS = 24
W_MNL = 0.7
TOP_K = 512
DAY = 86400
RECLICK_HOURS = 12.0


def read_any(data, name, **kw):
    pq, csv = data / (name + ".parquet"), data / (name + ".csv")
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            print("parquet read failed for", name, e)
    return pd.read_csv(csv, **kw)


def load_feedback(data):
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
    eclk = np.bincount(ev, weights=clk, minlength=E)
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

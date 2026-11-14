"""Team Nova - Phase 2 submission, round 2 candidate "model_angle2".

Same as model_angle (nova global pick + re-click at ANY lag), but when a user has
several previously clicked live items we take the one they clicked most often,
breaking ties by most recent click. model_angle / HIST=0 used most-recent only.

On the R1 pooled-slate scorer (r2/scorer.py, picks fit on pre-R1 feedback only)
this is 38.44 vs HIST=0 38.40 and nova_user 17.79. A second fold (train before
the last day of R1, test that day) is a bigger gap: 55.64 vs HIST=0 49.83, because
frequency only becomes informative once users have repeat clicks.

If we go past BUDGET_S seconds we fall back to HIST=0 (most recent live click).
If the MNL fit fails we use the smoothed CTR. Reads feedback_long.parquet/csv,
or feedback.jsonl if the long file is missing.
"""
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
RECLICK_HOURS = 1e9
RANK = "count"          # "count" = most clicks, tie recent; "recent" = HIST=0
BUDGET_S = 150.0
import time as _time
_T0 = [_time.time()]


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


def logger_slice(fb):
    # the random logger shows 3 items; later official feedback is pooled team slates (4-6 items)
    # and a plain item MNL on those slates credits habit re-clicks to the item. fit appeal on the
    # logger days only (mean slate length <= 3.5). fall back to all events if that looks too thin.
    try:
        n = fb.groupby("event_id").size()
        ts = fb.groupby("event_id")["timestamp"].first()
        day = ts // 86400
        mean_n = n.groupby(day).mean()
        good = mean_n[mean_n <= 3.5].index
        ev = day[day.isin(good)].index
        out = fb[fb["event_id"].isin(ev)]
        if out["event_id"].nunique() >= 1000:
            print("item scores on logger-like days:", int(out["event_id"].nunique()), "events (skipped pooled slates)")
            return out
    except Exception as e:
        print("logger slice failed, using all events:", e)
    return fb


def item_scores(fb, items):
    fb = fb.astype({"itemID": str, "user_id": str, "event_id": str})
    fb_fit = logger_slice(fb)
    ctr = smoothed_ctr(fb_fit, items)
    try:
        b = fit_mnl(fb_fit).reindex(items).fillna(0.0).to_numpy()
        s = W_MNL * b + (1 - W_MNL) * np.log(ctr)
        if not np.isfinite(s).all():
            raise ValueError("non finite mnl scores")
        return s
    except Exception as e:
        print("mnl failed, using plain ctr:", e)
        return ctr


def user_picks(fb, rows, vf_map, vt_map):
    # rowid -> a live item the user clicked before (any lag)
    try:
        fb = fb.astype({"user_id": str, "itemID": str})
        clk = fb[fb["clicked"].astype(int) == 1]
        g = clk.groupby(["user_id", "itemID"]).agg(nclk=("clicked", "size"), tc=("timestamp", "max")).reset_index()
        m = rows[["rowid", "user_id", "timestamp"]].merge(g, on="user_id")
        gap = m["timestamp"].astype("int64") - m["tc"].astype("int64")
        m = m[(gap >= 0) & (gap <= RECLICK_HOURS * 3600)].copy()
        m["vf"] = m["itemID"].map(vf_map)
        m["vt"] = m["itemID"].map(vt_map)
        t = m["timestamp"].astype("int64")
        m = m[m["vf"].notna() & (m["vf"] <= t) & (t < m["vt"])].copy()
        # time guard: fall back to HIST=0 (most recent)
        if RANK != "recent" and _time.time() - _T0[0] > BUDGET_S:
            print("over time budget, falling back to most-recent (HIST=0)")
            m["r"] = m["tc"].astype(float)
        elif RANK == "count":
            # most clicks, tie-break by most recent click
            m["r"] = m["nclk"].astype(float) * 1e12 + m["tc"].astype(float)
        else:
            m["r"] = m["tc"].astype(float)
        m = m.sort_values("r").drop_duplicates("rowid", keep="last")
        return dict(zip(m["rowid"], m["itemID"]))
    except Exception as e:
        print("user picks failed, global only:", e)
        return {}


def predict(data_dir, batch_size=256):
    _T0[0] = _time.time()
    data = Path(data_dir)
    rows = read_any(data, "x_pred", dtype={"rowid": str, "user_id": str})
    rows = rows.astype({"rowid": str, "user_id": str}).drop_duplicates("rowid")
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
    print("user picks:", len(upick), "rank:", RANK, "elapsed", f"{_time.time()-_T0[0]:.1f}s")

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
        out = []
        for a, b in zip(r, chosen):
            out.append((str(a), str(upick.get(a, b))))
        yield out

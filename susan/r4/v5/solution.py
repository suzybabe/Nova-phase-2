import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorer as nu
W_DEFAULT = np.array([1.103, 1.376, -0.374, -0.271])
POOLED_MAX_SLATE = 2.8
def _read_feedback(d):
    pq = d / "feedback_long.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq, columns=["event_id", "user_id", "timestamp", "itemID", "clicked"])
        except Exception as e:
            print("parquet failed:", e, file=sys.stderr)
    if (d / "feedback_long.csv").exists():
        return pd.read_csv(d / "feedback_long.csv")
    rows = []
    with open(d / "feedback.jsonl") as f:
        for line in f:
            e = json.loads(line)
            for it in e["displayed_items"]:
                rows.append((e["event_id"], e["user_id"], int(e["timestamp"]), it, int(it == e.get("clicked_item"))))
    return pd.DataFrame(rows, columns=["event_id", "user_id", "timestamp", "itemID", "clicked"])
def _read(d, name):
    p = d / (name + ".parquet")
    try:
        return pd.read_parquet(p) if p.exists() else pd.read_csv(d / (name + ".csv"))
    except Exception:
        return pd.read_csv(d / (name + ".csv"))
def _feats(ap, items, n, hrs):
    return np.c_[ap, np.log1p(n), np.log1p(np.maximum(hrs, 0)), (hrs < 12).astype(float)]
def fit_choice(fb, ap):
    try:
        day = fb.timestamp // 86400
        ev = fb.groupby("event_id").agg(n=("itemID", "size"), t=("timestamp", "first"))
        dsz = ev.groupby(ev.t // 86400).n.mean(); pdays = set(dsz[dsz < POOLED_MAX_SLATE].index)
        P = fb[day.isin(pdays)]
        P = P[P.event_id.isin(P.loc[P.clicked == 1, "event_id"])]
        cc = fb.loc[fb.clicked == 1, ["user_id", "itemID", "timestamp"]].rename(columns={"timestamp": "tc"})
        m = P[["event_id", "user_id", "timestamp", "itemID", "clicked"]].merge(cc, on=["user_id", "itemID"])
        m = m[m.tc < m.timestamp]
        m = m.groupby(["event_id", "itemID"]).agg(n=("tc", "size"), last=("tc", "max"), t=("timestamp", "first"), y=("clicked", "first")).reset_index()
        m = m[m.groupby("event_id").y.transform("sum") == 1]
        if m.event_id.nunique() < 1000:
            raise ValueError(f"only {m.event_id.nunique()} pooled training slates")
        F = _feats(ap.reindex(m.itemID).fillna(0).to_numpy(), None, m.n.to_numpy(), (m.t - m["last"]).to_numpy() / 3600)
        e = pd.factorize(m.event_id)[0]; y = m.y.to_numpy(float)
        def loss(w):
            u = F @ w; u = u - u.max(); ex = np.exp(u); den = np.bincount(e, weights=ex)
            f = -(y * u).sum() + np.log(den).sum() + 0.01 * w @ w
            g = -(F * y[:, None]).sum(0) + ((F * (ex / den[e])[:, None])).sum(0) + 0.02 * w
            return f, g
        w = minimize(loss, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B").x
        if not np.isfinite(w).all():
            raise ValueError("non-finite weights")
        print(f"[nova-r4-v5] choice model fitted on {e.max()+1:,} pooled slates, w={np.round(w, 3)}", file=sys.stderr)
        return w
    except Exception as ex:
        print("[nova-r4-v5] choice fit failed, default weights:", ex, file=sys.stderr)
        return W_DEFAULT
def predict(data_dir, batch_size=256):
    t0 = time.time(); d = Path(data_dir)
    fb = _read_feedback(d).astype({"event_id": str, "user_id": str, "itemID": str})
    fb["timestamp"] = fb.timestamp.astype(np.int64); fb["clicked"] = fb.clicked.astype(int)
    cat = _read(d, "catalog").astype({"itemID": str}).drop_duplicates("itemID").reset_index(drop=True)
    rows = _read(d, "x_pred").astype({"rowid": str, "user_id": str}).drop_duplicates("rowid").reset_index(drop=True)
    rows["timestamp"] = rows.timestamp.astype(np.int64)
    sc = nu.Scorer(w_knn=0.0, w_reclick=0.0).fit(fb, cat)
    ap = pd.Series(sc.ap_all, index=sc.items)
    W = fit_choice(fb, ap)
    vf = cat.valid_from.to_numpy(np.int64); vt = cat.valid_to.to_numpy(np.int64); citems = cat.itemID.to_numpy()
    vfm = pd.Series(vf, index=citems); vtm = pd.Series(vt, index=citems)
    h = fb[fb.clicked == 1].groupby(["user_id", "itemID"]).agg(n=("clicked", "size"), last=("timestamp", "max")).reset_index()
    q = rows[["rowid", "user_id", "timestamp"]].merge(h, on="user_id")
    q = q[(q.itemID.map(vfm) <= q.timestamp) & (q.timestamp < q.itemID.map(vtm))]
    pick = {}
    if len(q):
        q = q.assign(u=_feats(ap.reindex(q.itemID).fillna(0).to_numpy(), None, q.n.to_numpy(), (q.timestamp - q["last"]).to_numpy() / 3600) @ W)
        best = q.sort_values("u", ascending=False).drop_duplicates("rowid")
        pick = dict(zip(best.rowid, best.itemID))
    a = ap.reindex(citems).fillna(0.0).to_numpy(); order = np.argsort(-a, kind="stable")
    o_items, o_vf, o_vt = citems[order], vf[order], vt[order]
    rid = rows.rowid.to_numpy(); ts = rows.timestamp.to_numpy(); out = np.empty(len(rows), dtype=object)
    for s in range(0, len(rows), 8192):
        t = ts[s:s + 8192][:, None]; ok = (o_vf[None, :] <= t) & (t < o_vt[None, :])
        if not ok.any(1).all():
            raise RuntimeError("a row has no live item")
        out[s:s + 8192] = o_items[ok.argmax(1)]
    for i, r in enumerate(rid):
        if r in pick:
            out[i] = pick[r]
    print(f"[nova-r4-v5] done {time.time()-t0:.1f}s | personal picks {len(pick):,} of {len(rows):,} | distinct {len(set(out)):,}", file=sys.stderr, flush=True)
    for s in range(0, len(rid), batch_size):
        yield [(str(r), str(p)) for r, p in zip(rid[s:s + batch_size], out[s:s + batch_size])]

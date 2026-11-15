import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, "/home/user/susan_r4/nu"); import scorer as nu
D4 = "/home/user/data/round-4/phase-2-round-4-input"; CAT3 = "/home/user/data/round-3/phase-2-round-3-input/catalog.csv"
fl = pd.read_parquet(D4 + "/feedback_long.parquet").astype({"event_id": str, "user_id": str, "itemID": str})
fl["timestamp"] = fl.timestamp.astype(np.int64); fl["clicked"] = fl.clicked.astype(int)
cat = pd.read_csv(CAT3, dtype={"itemID": str}).drop_duplicates("itemID")
day = fl.timestamp // 86400; ev = fl.groupby("event_id").agg(n=("itemID", "size"), d=("timestamp", "first"))
dsz = ev.groupby(ev.d // 86400).n.mean(); pooled_days = set(dsz[dsz < 2.8].index)
last2 = sorted(day.unique())[-2:]; cut = last2[0] * 86400
tr = fl[fl.timestamp < cut]; te = fl[fl.timestamp >= cut]
print(f"pooled days {sorted(pooled_days)} | train rows {len(tr):,} | test pooled events {te.event_id.nunique():,}")
slate = te.groupby("event_id").itemID.apply(list); tclk = te[te.clicked == 1].drop_duplicates("event_id").set_index("event_id").itemID
req = te.drop_duplicates("event_id").set_index("event_id")[["user_id", "timestamp"]].loc[tclk.index]
def fit_ap(df):
    s = nu.Scorer(w_knn=0, w_reclick=1.5).fit(df, cat); return pd.Series(s.ap_all, index=s.items)
ap_all = fit_ap(tr); ap_rand = fit_ap(tr[~(tr.timestamp // 86400).isin(pooled_days)])
c = tr[tr.clicked == 1]
hist = c.groupby(["user_id", "itemID"]).agg(n=("clicked", "size"), last=("timestamp", "max")).reset_index()
vf = cat.set_index("itemID").valid_from; vt = cat.set_index("itemID").valid_to
def live_mask(items, t): return (vf.reindex(items).to_numpy() <= t) & (t < vt.reindex(items).to_numpy())
_ord = {}
def global_best(ap, t):
    k = id(ap)
    if k not in _ord:
        o = ap.sort_values(ascending=False).index[:500]; _ord[k] = (o, vf.reindex(o).to_numpy(), vt.reindex(o).to_numpy(), ap[o].to_numpy())
    o, a, b, v = _ord[k]; j = ((a <= t) & (t < b)).argmax(); return o[j], v[j]
q = req.reset_index().merge(hist, on="user_id"); q = q[live_mask(q.itemID, q.timestamp.to_numpy())].copy()
q["hrs"] = (q.timestamp - q["last"]) / 3600
P = tr[(tr.timestamp // 86400).isin(pooled_days)]; pc = P[P.clicked == 1].drop_duplicates("event_id").event_id
P = P[P.event_id.isin(pc)]
cc = c.rename(columns={"timestamp": "tc"})[["user_id", "itemID", "tc"]]
m = P[["event_id", "user_id", "timestamp", "itemID", "clicked"]].merge(cc, on=["user_id", "itemID"])
m = m[m.tc < m.timestamp].groupby(["event_id", "itemID"]).agg(n=("tc", "size"), last=("tc", "max"), t=("timestamp", "first"), y=("clicked", "first")).reset_index()
m["hrs"] = (m.t - m["last"]) / 3600
m = m[m.groupby("event_id").y.transform("sum") == 1]
def feats(df, ap): return np.c_[ap.reindex(df.itemID).fillna(0).to_numpy(), np.log1p(df.n), np.log1p(df.hrs), (df.hrs < 12).astype(float)]
e = pd.factorize(m.event_id)[0]; y = m.y.to_numpy()
def fit_logit(F):
    def loss(w):
        u = F @ w; mx = np.bincount(e, weights=np.exp(u)); return -(y * u).sum() + np.log(mx).sum() + 0.01 * w @ w
    return minimize(loss, np.zeros(F.shape[1]), method="L-BFGS-B").x
print(f"choice-model training: {e.max()+1:,} pooled slates where the click went to a past-clicked item")
W_all = fit_logit(feats(m, ap_all)); W_rand = fit_logit(feats(m, ap_rand))
print("learned weights [appeal, log clicks, log hours since last click, <12h]:", np.round(W_all, 3), "| random-day appeal:", np.round(W_rand, 3))
def evaluate(name, ap, rule, W=None):
    picks = {}
    gq = q.copy(); gq["ap"] = ap.reindex(gq.itemID).fillna(0).to_numpy()
    if W is not None: gq["u"] = feats(gq, ap) @ W
    best = gq.sort_values("u" if W is not None else "ap", ascending=False).drop_duplicates("event_id").set_index("event_id")
    for eid, (u, t) in req.iterrows():
        gi, gap = global_best(ap, t)
        if eid in best.index and (W is not None or best.at[eid, "ap"] + 1.5 > gap): picks[eid] = best.at[eid, "itemID"]
        else: picks[eid] = gi
    p = pd.Series(picks); inS = np.array([p[k] in slate[k] for k in p.index]); cc_ = tclk[inS]
    print(f"{name:<52} pick in slate {inS.mean():6.1%} | clicked scored {len(cc_):>6,} | SCORE {100*(p[cc_.index].to_numpy()==cc_.to_numpy()).mean():6.2f}", flush=True)
evaluate("(1) Nova User rule (appeal all days + 1.5 re-click)", ap_all, "nu")
evaluate("(2) Nova User rule, appeal from random days only", ap_rand, "nu")
evaluate("(3) learned re-click choice (appeal all days)", ap_all, "lc", W_all)
evaluate("(4) learned re-click choice (appeal random days)", ap_rand, "lc", W_rand)

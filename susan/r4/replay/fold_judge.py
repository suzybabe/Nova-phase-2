import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, "/home/user/susan_r4/nu"); import scorer as nu
D4 = "/home/user/data/round-4/phase-2-round-4-input"; CAT3 = "/home/user/data/round-3/phase-2-round-3-input/catalog.csv"
fl = pd.read_parquet(D4 + "/feedback_long.parquet").astype({"event_id": str, "user_id": str, "itemID": str})
fl["timestamp"] = fl.timestamp.astype(np.int64); fl["clicked"] = fl.clicked.astype(int)
cat = pd.read_csv(CAT3, dtype={"itemID": str}).drop_duplicates("itemID")
vf = cat.set_index("itemID").valid_from; vt = cat.set_index("itemID").valid_to
ev = fl.groupby("event_id").agg(n=("itemID", "size"), t=("timestamp", "first")); dsz = ev.groupby(ev.t // 86400).n.mean()
pdays = set(dsz[dsz < 2.8].index); fl["pooled"] = (fl.timestamp // 86400).isin(pdays)
days = sorted((fl.timestamp // 86400).unique())
def add_left(H):
    E = H.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]].sort_values(["user_id", "timestamp"])
    E["next_t"] = E.groupby("user_id").timestamp.shift(-1)
    H = H.merge(E[["event_id", "next_t"]], on="event_id", how="left")
    H["left"] = ((H.next_t - H.timestamp > 1800) | (H.next_t.isna() & (H.timestamp < H.timestamp.max() - 1800))).astype(float)
    return H
def stats(R, H):
    m = R[["key", "user_id", "itemID", "t"]].merge(H[["user_id", "itemID", "timestamp", "clicked", "pooled", "left", "next_t"]], on=["user_id", "itemID"])
    m = m[m.timestamp < m.t]; c = m[(m.clicked == 1)].copy(); c.loc[~(c.next_t < c.t) & ~(c.t - c.timestamp > 1800), "left"] = 0.0
    a = c.groupby("key").agg(n=("timestamp", "size"), last=("timestamp", "max"), first=("timestamp", "min"), npool=("pooled", "sum"), left=("left", "mean"))
    s = m[m.clicked == 0].groupby("key").size().rename("seen")
    out = R.set_index("key").join(a).join(s); out["seen"] = out.seen.fillna(0); return out[out.n.fillna(0) > 0].reset_index()
def F(df, ap, apr, ext):
    hrs = (df.t - df["last"]) / 3600; f = [ap.reindex(df.itemID).fillna(0).to_numpy(), np.log1p(df.n), np.log1p(hrs.clip(lower=0)), (hrs < 12).astype(float)]
    if ext: f += [df["left"].fillna(0).to_numpy(), np.log1p(df.seen)]
    return np.column_stack(f)
def fit(X, e, y):
    def loss(w):
        u = X @ w; u = u - u.max(); ex = np.exp(u); den = np.bincount(e, weights=ex)
        return -(y * u).sum() + np.log(den).sum() + 0.01 * w @ w, -(X * y[:, None]).sum(0) + (X * (ex / den[e])[:, None]).sum(0) + 0.02 * w
    return minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
for name, cd, ed in [("FOLD 1 (test 12-13 Jan)", days[-2], days[-1] + 1), ("FOLD 2 (test 11 Jan)", days[-3], days[-3] + 1)]:
    tr = add_left(fl[fl.timestamp < cd * 86400]); te = fl[(fl.timestamp >= cd * 86400) & (fl.timestamp < ed * 86400)]
    tclk = te[te.clicked == 1].drop_duplicates("event_id").set_index("event_id").itemID
    slate = te[te.event_id.isin(tclk.index)].groupby("event_id").itemID.apply(set)
    req = te.drop_duplicates("event_id").set_index("event_id").loc[tclk.index, ["user_id", "timestamp"]]
    s1 = nu.Scorer(w_knn=0, w_reclick=0).fit(tr, cat); ap = pd.Series(s1.ap_all, index=s1.items); apr = ap
    P = tr[tr.pooled & tr.event_id.isin(tr.loc[tr.pooled & (tr.clicked == 1), "event_id"])]
    TR = P[["event_id", "user_id", "itemID", "timestamp", "clicked"]].rename(columns={"timestamp": "t", "clicked": "y"}).reset_index(drop=True)
    TR["key"] = np.arange(len(TR)); TR = stats(TR, tr); TR = TR[TR.groupby("event_id").y.transform("sum") == 1]
    e = pd.factorize(TR.event_id)[0]; y = TR.y.to_numpy(float)
    c = tr[tr.clicked == 1][["user_id", "itemID"]].drop_duplicates()
    TE = req.reset_index().merge(c, on="user_id").rename(columns={"timestamp": "t"})
    TE = TE[(TE.itemID.map(vf) <= TE.t) & (TE.t < TE.itemID.map(vt))].reset_index(drop=True); TE["key"] = np.arange(len(TE)); TE = stats(TE, tr)
    o = ap.sort_values(ascending=False).index[:500]; ovf, ovt = vf.reindex(o).to_numpy(), vt.reindex(o).to_numpy(); t = req.timestamp.to_numpy()[:, None]
    gpick = pd.Series(o[((ovf[None, :] <= t) & (t < ovt[None, :])).argmax(1)], index=req.index)
    print(f"\n{name}: training slates {e.max()+1:,} | test clicked slates {len(tclk):,}", flush=True)
    picks = {}
    for lab, ext in [("BASE = v5", False), ("v5 + fatigue/session", True)]:
        w = fit(F(TR, ap, apr, ext), e, y); u = F(TE, ap, apr, ext) @ w
        best = TE.assign(u=u).sort_values("u", ascending=False).drop_duplicates("event_id").set_index("event_id").itemID
        p = gpick.copy(); p.update(best); picks[lab] = p
    nub = TE.assign(a=ap.reindex(TE.itemID).fillna(0).to_numpy()).sort_values("a", ascending=False).drop_duplicates("event_id").set_index("event_id")
    gap = ap.reindex(gpick).to_numpy(); p = gpick.copy(); k = nub.index
    p[k] = np.where(nub.a.to_numpy() + 1.5 > pd.Series(gap, index=gpick.index)[k].to_numpy(), nub.itemID.to_numpy(), gpick[k].to_numpy()); picks["Nova User rule"] = p
    J = te[te.event_id.isin(tclk.index)][["event_id", "user_id", "itemID"]].copy(); J["y"] = (J.itemID.to_numpy() == tclk.reindex(J.event_id).to_numpy()).astype(float)
    def habit(D):
        D = D.copy(); D["t"] = cd * 86400; D["key"] = np.arange(len(D)); S = stats(D, tr).set_index("key")
        n = S.n.reindex(D.key).fillna(0).to_numpy(); hrs = ((S.t - S["last"]) / 3600).reindex(D.key).fillna(1e4).to_numpy(); seen = S.seen.reindex(D.key).fillna(0).to_numpy()
        return np.c_[(n > 0).astype(float), np.log1p(n), np.log1p(hrs) * (n > 0), np.log1p(seen)]
    HJ = habit(J); items = pd.Index(J.itemID.unique()); ii = items.get_indexer(J.itemID); ej = pd.factorize(J.event_id)[0]; yj = J.y.to_numpy()
    NI = len(items)
    def jl(x):
        b, w = x[:NI], x[NI:]; u = b[ii] + HJ @ w; u = u - u.max(); ex = np.exp(u); den = np.bincount(ej, weights=ex); pr = ex / den[ej]
        f = -(yj * u).sum() + np.log(den).sum() + 0.5 * (b @ b)
        return f, np.r_[np.bincount(ii, weights=pr - yj, minlength=NI) + b, HJ.T @ (pr - yj)]
    x = minimize(jl, np.zeros(NI + HJ.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x
    bJ = pd.Series(x[:NI], index=items); wJ = x[NI:]; bdef = float(np.percentile(bJ, 25))
    print(f"  judge habit weights [clicked before, log clicks, log hrs, log seen]: {np.round(wJ, 2)}", flush=True)
    uS = pd.Series(bJ.reindex(J.itemID).to_numpy() + HJ @ wJ, index=J.event_id.to_numpy())
    denS = np.exp(uS).groupby(level=0).sum()
    for lab, p in picks.items():
        P = pd.DataFrame({"event_id": p.index, "user_id": req.user_id.reindex(p.index).to_numpy(), "itemID": p.to_numpy()})
        up = bJ.reindex(P.itemID).fillna(bdef).to_numpy() + habit(P) @ wJ
        inS = np.array([i in slate[k] for k, i in zip(P.event_id, P.itemID)])
        den = denS.reindex(P.event_id).to_numpy() + np.where(inS, 0.0, np.exp(up))
        real = (p[tclk.index[inS]].to_numpy() == tclk[inS].to_numpy()).mean()
        print(f"  {lab:<22} pick in slate {inS.mean():6.1%} | real-click score {100*real:6.2f} | JUDGE score (all rows) {100*np.mean(np.exp(up) / den):6.2f}", flush=True)

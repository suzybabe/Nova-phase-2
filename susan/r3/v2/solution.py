"""Team Nova - Round 3 solution v2: item-quality model (smoothed CTR) + co-click personalisation.

Why: on our validation (train 6-7 Jan, test 8 Jan random slates) plain item CTR ranks clicked items
better (hit@1 0.4127) than the organisers' SASRec starter (0.3955) or our v1 (0.3885).
Score(user, item) = log(smoothed CTR of item) + BETA * standardised co-click similarity to the user's past clicks.
Pick = best-scoring item that is available at the request time. numpy/pandas/scipy only, CPU, fast.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

PRIOR = 200.0       # CTR smoothing strength (impressions) - best on validation
HALF_LIFE_D = 1.0   # recency weighting of impressions (days) - helped on validation
BETA = 0.5          # weight of co-click personalisation - best on validation (eval_v2c hit@1 0.4267)
USER_CHUNK = 500    # users per block (keeps memory < ~1 GB)


def load(data_dir):
    data = Path(data_dir)
    fl = pd.read_parquet(data / "feedback_long.parquet") if (data / "feedback_long.parquet").exists() \
        else pd.read_csv(data / "feedback_long.csv")
    fl = fl[["event_id", "user_id", "timestamp", "itemID", "clicked"]].copy()
    fl["user_id"] = fl["user_id"].astype(str); fl["itemID"] = fl["itemID"].astype(str)
    fl["timestamp"] = fl["timestamp"].astype(np.int64); fl["clicked"] = fl["clicked"].astype(np.int64)
    cat = pd.read_csv(data / "catalog.csv", dtype={"itemID": str})
    xp = pd.read_csv(data / "x_pred.csv", dtype={"rowid": str, "user_id": str})
    xp["timestamp"] = xp["timestamp"].astype(np.int64)
    return fl, cat, xp


class ItemQuality:
    def __init__(self, prior=PRIOR, half_life_d=HALF_LIFE_D, beta=BETA):
        self.prior, self.hl, self.beta = prior, half_life_d, beta

    def fit(self, fl, cat):
        self.items = pd.Index(sorted(set(fl.itemID) | set(cat.itemID)))
        w = np.ones(len(fl)) if self.hl is None else \
            0.5 ** ((fl.timestamp.max() - fl.timestamp.to_numpy()) / (self.hl * 86400.0))
        ii = self.items.get_indexer(fl.itemID)
        imp = np.bincount(ii, weights=w, minlength=len(self.items))
        clk = np.bincount(ii, weights=w * fl.clicked.to_numpy(), minlength=len(self.items))
        p0 = clk.sum() / max(imp.sum(), 1e-9)
        self.logctr = np.log((clk + self.prior * p0) / (imp + self.prior))
        c = fl[fl.clicked == 1]
        self.users = pd.Index(sorted(set(c.user_id)))
        X = sp.csr_matrix((np.ones(len(c)), (self.users.get_indexer(c.user_id), self.items.get_indexer(c.itemID))),
                          shape=(len(self.users), len(self.items)))
        X.data[:] = 1.0; self.X = X
        C = (X.T @ X).tocsr(); C.setdiag(0); C.eliminate_zeros()
        deg = np.asarray(X.sum(0)).ravel() + 1e-9
        self.Cn = C.multiply(1 / np.sqrt(deg)[None, :]).tocsr()           # column-normalised co-clicks
        self.CnT = self.Cn.T.tocsr()
        # one fixed scale for co-click scores, measured on (user, shown item) pairs from training
        smp = fl.sample(min(len(fl), 200000), random_state=0)
        su = self.users.get_indexer(smp.user_id); si = self.items.get_indexer(smp.itemID)
        cf = self._cf_pairs(su, si)
        self.cf_sd = float(cf.std()) + 1e-9
        return self

    def _cf_pairs(self, u, i):
        """co-click score for (user index, item index) pairs; unknown user or item -> 0."""
        out = np.zeros(len(u))
        ok = (u >= 0) & (i >= 0)
        if ok.any():
            out[ok] = np.asarray(self.X[u[ok]].multiply(self.CnT[i[ok]]).sum(1)).ravel()
        return out

    def user_scores(self, uidx):
        """Dense (len(uidx), n_items) scores for known-user indices (-1 = unknown user)."""
        s = np.tile(self.logctr.astype(np.float32), (len(uidx), 1))
        if self.beta:
            ok = uidx >= 0
            if ok.any():
                cf = (self.X[uidx[ok]] @ self.Cn).toarray()
                s[ok] += (self.beta / self.cf_sd) * cf.astype(np.float32)
        return s

    def score_pairs(self, users, items):
        """users (Q,), items (Q,K) -> (Q,K) scores (for validation)."""
        u = self.users.get_indexer(pd.Index(users)); it = self.items.get_indexer(pd.Index(np.asarray(items).ravel())).reshape(np.shape(items))
        out = self.logctr[np.clip(it, 0, None)].astype(float)
        if self.beta:
            uu = np.repeat(u, it.shape[1]); cf = self._cf_pairs(uu, it.ravel()).reshape(it.shape)
            out += (self.beta / self.cf_sd) * cf
        return out

    def recommend(self, xp, cat):
        ci = self.items.get_indexer(cat.itemID)                 # catalogue columns
        vf, vt = cat.valid_from.to_numpy(np.int64), cat.valid_to.to_numpy(np.int64)
        cat_items = cat.itemID.to_numpy()
        recs = np.empty(len(xp), dtype=object)
        xs = xp.assign(_r=np.arange(len(xp))).sort_values("user_id")
        ulist = xs.user_id.unique()
        for s in range(0, len(ulist), USER_CHUNK):
            us = ulist[s:s + USER_CHUNK]
            S = self.user_scores(self.users.get_indexer(pd.Index(us)))[:, ci]     # (U, n_cat)
            pos = pd.Series(np.arange(len(us)), index=us)
            part = xs[xs.user_id.isin(us)]
            rows = part._r.to_numpy(); t = part.timestamp.to_numpy()[:, None]
            sc = S[pos.reindex(part.user_id).to_numpy()]
            sc[~((vf[None, :] <= t) & (t < vt[None, :]))] = -np.inf
            best = sc.argmax(1)
            dead = ~np.isfinite(sc[np.arange(len(best)), best])
            recs[rows] = cat_items[best]
            if dead.any():                                          # nothing available: first live / first item
                for j in np.flatnonzero(dead):
                    live = np.flatnonzero((vf <= t[j, 0]) & (t[j, 0] < vt))
                    recs[rows[j]] = cat_items[live[0] if len(live) else 0]
        return recs


def predict(data_dir, batch_size=256):
    t0 = time.time()
    fl, cat, xp = load(data_dir)
    xp = xp.drop_duplicates("rowid").reset_index(drop=True)
    m = ItemQuality().fit(fl, cat)
    recs = m.recommend(xp, cat)
    print(f"[nova-r3-v2] done {time.time()-t0:.1f}s, distinct items {len(set(recs))}", file=sys.stderr, flush=True)
    out = []
    for rid, it in zip(xp.rowid.to_numpy(), recs):
        out.append((str(rid), str(it)))
        if len(out) >= batch_size:
            yield out; out = []
    if out:
        yield out

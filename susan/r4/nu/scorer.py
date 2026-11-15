"""Nova User scorer (R3/R4): global appeal + personal taste + own-click bonus. numpy/pandas only.

score(u, i) = log appeal(i) + W_KNN * knn(u, i) / sd + W_RECLICK * clicked_before(u, i) + W_SNC * shown_never_clicked(u, i)
  appeal  : k-adjusted O/E appeal (Nova Data): f(k) = impression CTR at the user's EVENT INDEX k (sequence position,
            not hour); appeal_i = (clicks_i + 20) / (sum over impressions of f(k) + 20), from all history, no decay.
  knn     : item-item co-click taste: sum over the user's clicked items j of cos(j, i), cos from the
            user x item click matrix normalised by 1/sqrt(deg_u deg_i). dense float32 item x item (~7k^2).
  clicked_before: R3 users rarely get re-shown an item, but when a clicked item comes back its CTR is
            0.0565 vs 0.0075 fresh (late events, R3 history), so a re-click bonus is a real taste signal.
Weights were picked on fold A (history day 0, test day 1) and checked on fold B (history days 0-1, test day 2);
see /workspace/techarena/r34_user/README.md.
"""
import numpy as np
import pandas as pd

W_KNN = 0.0
W_RECLICK = 1.5
W_SNC = 0.0
EARLY_N = 10      # unused (kept for the constructor signature)
OE_S = 20.0


def _idx(col, index):
    """index.get_indexer for a column, cheap for categoricals (no per-row str copies)."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        m = index.get_indexer(col.cat.categories.astype(str))
        codes = col.cat.codes.to_numpy()
        return np.where(codes >= 0, m[np.maximum(codes, 0)], -1)
    return index.get_indexer(col.astype(str))


def _fact(col):
    if isinstance(col.dtype, pd.CategoricalDtype):
        codes, uniq = pd.factorize(col.cat.codes.to_numpy())
        return codes, pd.Index(col.cat.categories.astype(str)[uniq])
    return pd.factorize(col.astype(str))


class Scorer:
    name = "nova_user"

    def __init__(self, w_knn=W_KNN, w_reclick=W_RECLICK, w_snc=W_SNC, early_n=EARLY_N):
        self.w_knn, self.w_reclick, self.w_snc, self.early_n = w_knn, w_reclick, w_snc, early_n

    # ---------------- fit ----------------
    def fit(self, train_df, catalog_df):
        t = train_df
        ti = t["itemID"]
        tcat = ti.cat.categories.astype(str).to_numpy() if isinstance(ti.dtype, pd.CategoricalDtype) else pd.unique(ti.astype(str))
        items = pd.Index(pd.unique(np.concatenate([catalog_df["itemID"].astype(str).to_numpy(), np.asarray(tcat, dtype=object)])))
        self.items = items
        ic = _idx(ti, items)
        uc, users = _fact(t["user_id"])
        self.users = pd.Index(users)
        ts = t["timestamp"].to_numpy("int64")
        clk = t["clicked"].to_numpy().astype(np.int8) == 1
        NI, NU = len(items), len(users)
        # event index per user (sequence position)
        if "event_id" in t.columns:
            ec, _ = _fact(t["event_id"])
            ev = pd.DataFrame({"e": ec, "u": uc, "t": ts}).drop_duplicates("e")
            ev = ev.sort_values(["u", "t"], kind="stable")
            ev["k"] = ev.groupby("u").cumcount()
            kk = np.zeros(ec.max() + 1, np.int64); kk[ev.e.to_numpy()] = ev.k.to_numpy()
            eidx = kk[ec]
            self.n_ev = np.bincount(ev.u.to_numpy(), minlength=NU)
        else:
            o = np.lexsort((ts, uc)); eidx = np.empty(len(t), np.int64)
            eidx[o] = pd.Series(uc[o]).groupby(uc[o]).cumcount().to_numpy()
            self.n_ev = np.bincount(uc, minlength=NU)
        self.last_ts = pd.Series(ts).groupby(uc).max().reindex(range(NU)).to_numpy()
        # k-adjusted appeal (Nova Data finding 1-2): CTR depends on the user's event index k times item appeal.
        # f(k) = impression CTR at event index k; appeal_i = (clicks_i + S) / (sum_impr f(k) + S)  (O/E, shrunk to 1)
        K = int(eidx.max()) + 1
        fk = np.bincount(eidx, weights=clk, minlength=K) / np.maximum(np.bincount(eidx, minlength=K), 1)
        obs = np.bincount(ic, weights=clk, minlength=NI); exp_ = np.bincount(ic, weights=fk[eidx], minlength=NI)
        self.fk = fk
        self.ap_all = np.log((obs + OE_S) / (exp_ + OE_S))
        self.ap_cold = 0.0            # unseen item = prior mean (O/E = 1)
        # clicks / shown sets as sorted int64 keys
        key = uc.astype(np.int64) * NI + ic
        self.k_clk = np.unique(key[clk])
        shown = np.unique(key)
        self.k_snc = np.setdiff1d(shown, self.k_clk, assume_unique=True)
        # item-item co-click cosine (dense float32)
        cu, ci = uc[clk], ic[clk]
        pair = np.unique(cu.astype(np.int64) * NI + ci); cu, ci = pair // NI, pair % NI
        du = np.bincount(cu, minlength=NU).astype(np.float32); di = np.bincount(ci, minlength=NI).astype(np.float32)
        w = 1.0 / np.sqrt(du[cu] * di[ci])
        C = np.zeros((NI, NI), np.float32)
        # sum over users of w_ui * w_uj for item pairs: group clicks by user
        o = np.argsort(cu, kind="stable"); cu, ci, w = cu[o], ci[o], w[o]
        starts = np.r_[0, np.flatnonzero(np.diff(cu)) + 1]; lens = np.diff(np.r_[starts, len(cu)])
        for L in np.unique(lens):
            if L < 2:
                continue
            s = starts[lens == L]
            I = ci[s[:, None] + np.arange(L)]; Wt = w[s[:, None] + np.arange(L)]
            a = np.repeat(I, L, axis=1).ravel(); b = np.tile(I, (1, L)).ravel()
            v = (np.repeat(Wt, L, axis=1) * np.tile(Wt, (1, L))).ravel()
            np.add.at(C, (a, b), v.astype(np.float32))
        np.fill_diagonal(C, 0.0)
        self.C = C
        # user click lists (CSR-ish) for the knn sum
        self.u_ptr = np.r_[0, np.cumsum(np.bincount(cu, minlength=NU))]
        self.u_items = ci
        self.u_w = w                       # 1/sqrt(deg_u deg_i) per (user, clicked item), same order as u_items
        # scale of knn over (user, random item) so W_KNN is in sd units
        rng = np.random.default_rng(0)
        su = rng.choice(NU, size=min(500, NU), replace=False)
        M = self._knn_dense(su)
        self.knn_sd = float(M.std()) if M.size and M.std() > 0 else 1.0
        return self

    def _knn_dense(self, uidx):
        """dense (len(uidx) x NI) knn sums for user codes uidx (>=0)."""
        out = np.zeros((len(uidx), len(self.items)), np.float32)
        for r, u in enumerate(uidx):
            a, b = self.u_ptr[u], self.u_ptr[u + 1]
            js = self.u_items[a:b]
            if len(js):
                out[r] = self.C[js].sum(0)
                if len(js) > 1:
                    # leave-user-out: remove this user's own contribution to C[j, i] for her own clicked items
                    wj = self.u_w[a:b]
                    out[r, js] -= wj * (wj.sum() - wj)
        return out

    # ---------------- score ----------------
    def components(self, pairs_df, with_knn=None):
        with_knn = (self.w_knn != 0) if with_knn is None else with_knn
        p = pairs_df
        NI = len(self.items)
        ui = self.users.get_indexer(p["user_id"].astype(str)); ii = self.items.get_indexer(p["itemID"].astype(str))
        n = len(p); has_u = ui >= 0; has_i = ii >= 0
        # sequence position is NOT used for the pick: Nova Data found within-slate choice does not change with k
        # (it only scales click probability), so k is used in fit() to de-bias appeal, not here.
        ap = np.full(n, self.ap_cold)
        ap[has_i] = self.ap_all[ii[has_i]]
        key = np.where(has_u & has_i, ui.astype(np.int64) * NI + ii, -1)
        rc = np.isin(key, self.k_clk).astype(np.float64); snc = np.isin(key, self.k_snc).astype(np.float64)
        knn = np.zeros(n)
        sel = np.flatnonzero(has_u & has_i)
        if len(sel) and with_knn:
            us = ui[sel]; o = np.argsort(us, kind="stable"); sel, us = sel[o], us[o]
            uq, st = np.unique(us, return_index=True); en = np.r_[st[1:], len(us)]
            CH = 512
            for a in range(0, len(uq), CH):
                M = self._knn_dense(uq[a:a + CH])
                lo, hi = st[a], en[min(a + CH, len(uq)) - 1]
                loc = np.searchsorted(uq[a:a + CH], us[lo:hi])
                knn[sel[lo:hi]] = M[loc, ii[sel[lo:hi]]]
        return dict(appeal=ap, knn=knn / self.knn_sd, reclick=rc, snc=snc)

    def combine(self, c, w_knn=None, w_reclick=None, w_snc=None):
        w_knn = self.w_knn if w_knn is None else w_knn
        w_reclick = self.w_reclick if w_reclick is None else w_reclick
        w_snc = self.w_snc if w_snc is None else w_snc
        s = c["appeal"] + w_knn * c["knn"] + w_reclick * c["reclick"] + w_snc * c["snc"]
        return np.nan_to_num(np.asarray(s, np.float64), nan=self.ap_cold, posinf=0.0, neginf=self.ap_cold - 10)

    def score(self, pairs_df):
        return self.combine(self.components(pairs_df))

    # ---------------- candidates ----------------
    def candidates(self, rows_df, k=20, catalog_df=None):
        """per row: top-k items by (appeal + reclick bonus) among the top-3k appeal items and the user's own
        clicked items; live at the row's ts if catalog_df is given. (knn term ignored here; it is 0 by default.)"""
        r = rows_df[["rowid", "user_id", "timestamp"]].astype({"rowid": str, "user_id": str}).reset_index(drop=True)
        ui = self.users.get_indexer(r.user_id)
        top = np.argsort(-self.ap_all)[:3 * k]
        n = len(r)
        ri = [np.repeat(np.arange(n), len(top))]; it = [np.tile(top, n)]
        has = np.flatnonzero(ui >= 0)
        cnt = (self.u_ptr[ui[has] + 1] - self.u_ptr[ui[has]])
        ri.append(np.repeat(has, cnt))
        starts = np.repeat(self.u_ptr[ui[has]], cnt); off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        it.append(self.u_items[starts + off])
        ri = np.concatenate(ri); it = np.concatenate(it)
        own = np.r_[np.zeros(n * len(top), bool), np.ones(len(ri) - n * len(top), bool)]
        d = pd.DataFrame({"r": ri, "i": it, "own": own}).groupby(["r", "i"], as_index=False).own.max()
        d["s"] = self.ap_all[d.i.to_numpy()] + self.w_reclick * d.own.to_numpy()
        if catalog_df is not None:
            cat = catalog_df.set_index(catalog_df.itemID.astype(str))
            names = self.items[d.i.to_numpy()]
            vf = cat.valid_from.reindex(names).to_numpy(); vt = cat.valid_to.reindex(names).to_numpy()
            t = r.timestamp.to_numpy("int64")[d.r.to_numpy()]
            d = d[(vf <= t) & (t < vt)]
        d = d.sort_values(["r", "s"], ascending=[True, False]).groupby("r").head(k)
        return pd.DataFrame({"rowid": r.rowid.to_numpy()[d.r.to_numpy()], "itemID": self.items[d.i.to_numpy()]})


def make_scorer():
    return Scorer()

"""Adapter: makes the CF team's cf.py models follow the validation contract."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cf import ItemItemCF, ImplicitALS

DAY = 86_400


class CFModel:
    def __init__(self, kind="als", w=1.0, half_life_days=1.0, prior_strength=100.0,
                 factors=32, reg=0.1, alpha=20.0, iters=10):
        self.kind, self.w = kind, float(w)
        self.hl, self.k = half_life_days, float(prior_strength)
        self.als_params = dict(factors=factors, reg=reg, alpha=alpha, iters=iters)

    def fit(self, train, catalog):
        age = (train["timestamp"].max() - train["timestamp"]) / DAY
        wt = np.ones(len(train)) if self.hl is None else 0.5 ** (age / self.hl)
        g = pd.DataFrame({"i": train["itemID"], "w": wt, "wc": wt * train["clicked"]}).groupby("i")[["w", "wc"]].sum()
        self.gctr = g["wc"].sum() / g["w"].sum()
        self.ctr = (g["wc"] + self.k * self.gctr) / (g["w"] + self.k)
        items = pd.unique(pd.concat([train["itemID"], catalog["itemID"]]))
        users = pd.unique(train["user_id"])
        self.imap = pd.Series(np.arange(len(items)), index=items)
        self.umap = pd.Series(np.arange(len(users)), index=users)
        c = train[train["clicked"] == 1]
        R = sp.csr_matrix((np.ones(len(c)), (self.umap[c["user_id"]].to_numpy(), self.imap[c["itemID"]].to_numpy())),
                          shape=(len(users), len(items)))
        self.m = ItemItemCF().fit(R) if self.kind == "itemitem" else ImplicitALS(**self.als_params).fit(R)
        return self

    def score(self, pairs):
        base = pairs["itemID"].map(self.ctr).fillna(self.gctr).to_numpy(float)
        u = pairs["user_id"].map(self.umap)
        i = pairs["itemID"].map(self.imap)
        ok = (u.notna() & i.notna()).to_numpy()
        cf = np.zeros(len(pairs))
        if ok.any():
            uu = u[ok].astype(int).to_numpy(); ii = i[ok].astype(int).to_numpy()
            if self.kind == "itemitem":
                Ru = self.m.R[uu]
                cf[ok] = np.asarray(Ru.multiply(self.m.sim[:, ii].T).sum(1)).ravel()
            else:
                cf[ok] = np.einsum("ij,ij->i", self.m.X[uu], self.m.Y[ii])
        return cf + self.w * base

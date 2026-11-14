
"""Time-decayed ItemItemCF experiment.
Keeps original cf.py untouched.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

DAY = 86400


class TimeWeightedItemItemCF:
    def __init__(self, half_life_days=1.0):
        self.hl = float(half_life_days)

    def fit(self, train, catalog):
        items = pd.unique(pd.concat([train.itemID, catalog.itemID]))
        users = pd.unique(train.user_id)

        self.imap = pd.Series(np.arange(len(items)), index=items)
        self.umap = pd.Series(np.arange(len(users)), index=users)

        # Clicked interactions only
        c = train[train.clicked == 1].copy()

        # Recent clicks receive larger weights
        age = (train.timestamp.max() - c.timestamp) / DAY
        weights = 0.5 ** (age / self.hl)

        rows = self.umap[c.user_id].to_numpy()
        cols = self.imap[c.itemID].to_numpy()

        self.R = sp.csr_matrix(
            (weights, (rows, cols)),
            shape=(len(users), len(items)),
            dtype=np.float32,
        )

        # Weighted cosine similarity
        co = (self.R.T @ self.R).toarray()
        norms = np.sqrt(np.diag(co)) + 1e-9

        self.sim = co / norms[:, None] / norms[None, :]
        np.fill_diagonal(self.sim, 0)

        return self

    def score_cf(self, pairs):
        u = pairs.user_id.map(self.umap)
        i = pairs.itemID.map(self.imap)

        ok = (u.notna() & i.notna()).to_numpy()
        scores = np.zeros(len(pairs), dtype=np.float32)

        if ok.any():
            uu = u[ok].astype(int).to_numpy()
            ii = i[ok].astype(int).to_numpy()

            Ru = self.R[uu]
            scores[ok] = np.asarray(
                Ru.multiply(self.sim[:, ii].T).sum(1)
            ).ravel()

        return scores

import numpy as np
import pandas as pd
import scipy.sparse as sp


def make_synthetic(n_users=500, n_items=100, n_events=20000, slate=5, seed=0):
    """Fake data in the same shape as the real feedback: slates + one click or none."""
    rng = np.random.default_rng(seed)
    taste = rng.normal(size=(n_users, 8))
    item_f = rng.normal(size=(n_items, 8))
    rows = []
    for t in range(n_events):
        u = rng.integers(n_users)
        shown = rng.choice(n_items, slate, replace=False)
        p = np.exp(taste[u] @ item_f[shown].T)
        p = np.append(p, 3.0)                      # last slot = "no click"
        k = rng.choice(slate + 1, p=p / p.sum())
        rows.append((u, t, shown.tolist(), None if k == slate else int(shown[k])))
    return pd.DataFrame(rows, columns=["user_id", "timestamp", "displayed_items", "clicked_item"])


def build_matrix(events, n_users, n_items):
    """Users x items matrix of click counts."""
    ev = events.dropna(subset=["clicked_item"])
    r = ev["user_id"].to_numpy()
    c = ev["clicked_item"].astype(int).to_numpy()
    return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n_users, n_items))


class ItemItemCF:
    """Items are similar if the same users clicked both (cosine similarity)."""
    def fit(self, R):
        B = (R > 0).astype(np.float32).tocsc()
        norms = np.sqrt(np.asarray(B.sum(0)).ravel()) + 1e-9
        co = (B.T @ B).toarray()
        self.sim = co / norms[:, None] / norms[None, :]
        np.fill_diagonal(self.sim, 0)
        self.R = R.tocsr()
        return self

    def score(self, user_idx):
        return np.asarray(self.R[user_idx].toarray()) @ self.sim


class ImplicitALS:
    """Implicit-feedback ALS (Hu, Koren, Volinsky). Small and dependency-free."""
    def __init__(self, factors=32, reg=0.1, alpha=20.0, iters=10, seed=0):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve(self, R, Y):
        YtY = Y.T @ Y
        out = np.zeros((R.shape[0], self.f))
        eye = self.reg * np.eye(self.f)
        for u in range(R.shape[0]):
            s, e = R.indptr[u], R.indptr[u + 1]
            if s == e:
                continue
            Yi = Y[R.indices[s:e]]
            c = self.alpha * R.data[s:e]
            A = YtY + Yi.T @ (c[:, None] * Yi) + eye
            b = Yi.T @ (1 + c)
            out[u] = np.linalg.solve(A, b)
        return out

    def fit(self, R):
        R = R.tocsr()
        rng = np.random.default_rng(self.seed)
        self.Y = rng.normal(scale=0.1, size=(R.shape[1], self.f))
        RT = R.T.tocsr()
        for _ in range(self.iters):
            self.X = self._solve(R, self.Y)
            self.Y = self._solve(RT, self.X)
        self.X = self._solve(R, self.Y)
        return self

    def score(self, user_idx):
        return self.X[user_idx] @ self.Y.T


def recommend(model, R, user_ids, timestamps, valid_from, valid_to, pop):
    """One best available item per request. Unknown users get popularity."""
    known = np.asarray(user_ids) < R.shape[0]        # real data needs a user_id -> index map
    S = np.tile(pop, (len(user_ids), 1)).astype(float)
    if known.any():
        S[known] = model.score(np.asarray(user_ids)[known]) + 1e-6 * pop
    ok = (valid_from[None, :] <= np.asarray(timestamps)[:, None]) & \
         (np.asarray(timestamps)[:, None] < valid_to[None, :])
    S[~ok] = -np.inf
    return S.argmax(1)


if __name__ == "__main__":
    n_users, n_items = 500, 100
    ev = make_synthetic(n_users, n_items)
    split = int(len(ev) * 0.8)
    train, test = ev.iloc[:split], ev.iloc[split:]
    R = build_matrix(train, n_users, n_items)
    pop = np.asarray(R.sum(0)).ravel()

    test = test.dropna(subset=["clicked_item"])
    vf = np.zeros(n_items)
    vt = np.full(n_items, 1e12)
    for name, m in [("popularity", None), ("item-item", ItemItemCF().fit(R)),
                    ("ALS", ImplicitALS().fit(R))]:
        if m is None:
            pick = np.full(len(test), pop.argmax())
        else:
            pick = recommend(m, R, test.user_id.to_numpy(), test.timestamp.to_numpy(), vf, vt, pop)
        hit = np.mean(pick == test.clicked_item.astype(int).to_numpy())
        print(f"{name:12s} hit rate: {hit:.3f}")
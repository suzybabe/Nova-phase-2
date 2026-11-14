import numpy as np
import pandas as pd
from scipy.optimize import minimize

DAY = 86_400


class ChoiceLogit:
    """Learns P(item chosen | user clicks on the slate): softmax over the shown items.
    dim=0: one appeal score per item. dim>0: adds user-taste x item factors."""

    def __init__(self, dim=0, reg_item=1.0, reg_factor=5.0, half_life_days=None, max_iter=200, seed=0):
        self.dim = int(dim)
        self.reg_item = float(reg_item)
        self.reg_factor = float(reg_factor)
        self.half_life_days = half_life_days
        self.max_iter = int(max_iter)
        self.seed = seed

    def fit(self, train, catalog):
        n_click = train.groupby("event_id")["clicked"].transform("sum")
        t = train[n_click == 1]
        self.item_index = {it: i for i, it in enumerate(pd.unique(pd.concat([t["itemID"], catalog["itemID"]])))}
        self.user_index = {u: i for i, u in enumerate(pd.unique(t["user_id"]))}
        n_items, n_users, k = len(self.item_index), len(self.user_index), self.dim
        ev, _ = pd.factorize(t["event_id"])
        it = t["itemID"].map(self.item_index).to_numpy()
        us = t["user_id"].map(self.user_index).to_numpy()
        chosen = t["clicked"].to_numpy().astype(bool)
        n_ev = ev.max() + 1
        if self.half_life_days:
            age = (t["timestamp"].max() - t["timestamp"].to_numpy()) / DAY
            w_row = 0.5 ** (age / float(self.half_life_days))
        else:
            w_row = np.ones(len(t))
        w_ev = np.zeros(n_ev)
        w_ev[ev[chosen]] = w_row[chosen]
        rng = np.random.default_rng(self.seed)
        x0 = np.concatenate([np.zeros(n_items), 0.01 * rng.standard_normal(n_users * k),
                             0.01 * rng.standard_normal(n_items * k)])

        def unpack(x):
            return (x[:n_items], x[n_items:n_items + n_users * k].reshape(n_users, k),
                    x[n_items + n_users * k:].reshape(n_items, k))

        def loss_grad(x):
            b, P, Q = unpack(x)
            u = b[it] + (np.einsum("ij,ij->i", P[us], Q[it]) if k else 0.0)
            m = np.full(n_ev, -np.inf)
            np.maximum.at(m, ev, u)
            e = np.exp(u - m[ev])
            s = np.bincount(ev, weights=e, minlength=n_ev)
            p = e / s[ev]
            lse = np.log(s) + m
            nll = -(w_ev[ev[chosen]] * (u[chosen] - lse[ev[chosen]])).sum()
            g_u = w_ev[ev] * (p - chosen)
            g_b = np.bincount(it, weights=g_u, minlength=n_items) + 2 * self.reg_item * b
            loss = nll + self.reg_item * (b @ b)
            if k:
                g_P = np.zeros_like(P); g_Q = np.zeros_like(Q)
                np.add.at(g_P, us, g_u[:, None] * Q[it])
                np.add.at(g_Q, it, g_u[:, None] * P[us])
                g_P += 2 * self.reg_factor * P; g_Q += 2 * self.reg_factor * Q
                loss += self.reg_factor * ((P * P).sum() + (Q * Q).sum())
                return loss, np.concatenate([g_b, g_P.ravel(), g_Q.ravel()])
            return loss, g_b

        res = minimize(loss_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.b, self.P, self.Q = unpack(res.x)
        return self

    def score(self, pairs):
        ii = pairs["itemID"].map(self.item_index)
        known_item = ii.notna().to_numpy()
        ii = ii.fillna(0).astype(int).to_numpy()
        s = np.where(known_item, self.b[ii], 0.0)
        if self.dim:
            uu = pairs["user_id"].map(self.user_index)
            known_user = uu.notna().to_numpy() & known_item
            uu = uu.fillna(0).astype(int).to_numpy()
            s = s + np.where(known_user, np.einsum("ij,ij->i", self.P[uu], self.Q[ii]), 0.0)
        return s

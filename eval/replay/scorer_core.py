"""Copy of HabitIndex + ChoiceModel from r2/vm/scorer.py (Nova Rules calibrated pooled-slate scorer)."""
import numpy as np, pandas as pd, torch

class HabitIndex:
    """prior clicks of (user,item) strictly before t, from a feedback frame."""
    def __init__(self, fb):
        c = fb[fb.clicked_item.notna()][['user_id', 'clicked_item', 'timestamp']].sort_values('timestamp')
        self.g = {k: v.timestamp.values for k, v in c.groupby(['user_id', 'clicked_item'])}
    def feats(self, users, items, ts):
        n = len(users); F = np.zeros((n, 4), np.float32)
        for r, (u, i, t) in enumerate(zip(users, items, ts)):
            a = self.g.get((u, i))
            if a is None: continue
            k = np.searchsorted(a, t)
            if k == 0: continue
            dt = (t - a[k - 1]) / 3600.0
            F[r] = (1.0, np.log1p(k), np.log1p(dt), float(dt < 12))
        return F

class ChoiceModel:
    def __init__(self, lam=1.0, iters=300):
        self.lam, self.iters = lam, iters
    def fit(self, ev, hab):
        """ev: events (pooled slates) with a click. hab: HabitIndex over all feedback."""
        lg = ev[['event_id', 'user_id', 'timestamp', 'displayed_items', 'clicked_item']].explode('displayed_items')
        self.items = pd.Index(lg.displayed_items.unique())
        e = pd.factorize(lg.event_id)[0]; k = lg.groupby(e).cumcount().values; E, K = e.max() + 1, k.max() + 1
        F = hab.feats(lg.user_id.values, lg.displayed_items.values, lg.timestamp.values)
        IT = np.zeros((E, K), np.int64); M = np.zeros((E, K), bool); X = np.zeros((E, K, 4), np.float32); Y = np.zeros((E, K), np.float32)
        IT[e, k] = self.items.get_indexer(lg.displayed_items); M[e, k] = True; X[e, k] = F; Y[e, k] = (lg.displayed_items == lg.clicked_item).values
        IT, M, X, Y = map(torch.tensor, (IT, M, X, Y))
        b = torch.zeros(len(self.items), requires_grad=True); w = torch.zeros(4, requires_grad=True)
        opt = torch.optim.LBFGS([b, w], max_iter=self.iters, line_search_fn='strong_wolfe')
        def clos():
            opt.zero_grad()
            s = torch.where(M, b[IT] + (X * w).sum(-1), torch.tensor(-1e4))
            loss = -(torch.log_softmax(s, 1) * Y).sum() / E + self.lam * (b ** 2).sum() / E
            loss.backward(); return loss
        opt.step(clos)
        self.b = b.detach().numpy().astype(np.float64); self.w = w.detach().numpy().astype(np.float64)
        self.bdef = float(np.percentile(self.b, 25))   # unseen items: below-median appeal
        return self
    def util(self, users, items, ts, hab):
        idx = self.items.get_indexer(items); b = np.where(idx >= 0, self.b[np.maximum(idx, 0)], self.bdef)
        return b + hab.feats(users, items, ts) @ self.w
    def loglik(self, ev, hab):
        lg = ev[['event_id', 'user_id', 'timestamp', 'displayed_items', 'clicked_item']].explode('displayed_items')
        lg['u'] = self.util(lg.user_id.values, lg.displayed_items.values, lg.timestamp.values, hab)
        lse = lg.groupby('event_id').u.transform(lambda s: np.log(np.exp(s - s.max()).sum()) + s.max())
        return float((lg.u - lse)[lg.displayed_items == lg.clicked_item].mean())

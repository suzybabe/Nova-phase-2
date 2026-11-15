"""Which simple signals really predict clicks in Round 3? (numpy/pandas/scipy only, ~1-2 min on the VM)
Same split as r3_eval: train 6-7 Jan, test = 8 Jan slates with one click. Metric: slate hit@1 (ties count against).
Run ON THE VM:  /opt/techarena/venvs/torch/bin/python signals.py ~/data/round-3/phase-2-round-3-input
"""
import sys
import numpy as np, pandas as pd, scipy.sparse as sp

D = sys.argv[1].rstrip('/') + '/'
fl = pd.read_parquet(D + 'feedback_long.parquet')[['event_id', 'user_id', 'timestamp', 'itemID', 'clicked']]
cut = int(fl.timestamp.max()) // 86400 * 86400
tr, te = fl[fl.timestamp < cut], fl[fl.timestamp >= cut]
ec = te.groupby('event_id').clicked.sum(); te = te[te.event_id.isin(ec[ec == 1].index)].copy()
hour = lambda t: (t // 3600) % 24
g = tr.groupby('itemID').clicked.agg(['sum', 'size']); p0 = tr.clicked.mean()
ctr = (g['sum'] + 50 * p0) / (g['size'] + 50 * 1.0)
tr4 = tr.assign(hb=hour(tr.timestamp) // 4)
gh = tr4.groupby(['itemID', 'hb']).clicked.agg(['sum', 'size'])
ctr_h = (gh['sum'] + 50 * ctr.reindex(gh.index.get_level_values(0)).to_numpy()) / (gh['size'] + 50)
clk = tr[tr.clicked == 1].sort_values(['user_id', 'timestamp'])
prev = clk.groupby('user_id').itemID.shift()
trans = pd.DataFrame({'a': prev, 'b': clk.itemID}).dropna().groupby(['a', 'b']).size()
last = clk.groupby('user_id').itemID.last()
last3 = clk.groupby('user_id').itemID.apply(lambda s: list(s.iloc[-3:]))
items = pd.Index(sorted(set(fl.itemID))); users = pd.Index(sorted(set(clk.user_id)))
X = sp.csr_matrix((np.ones(len(clk)), (users.get_indexer(clk.user_id), items.get_indexer(clk.itemID))), shape=(len(users), len(items)))
X.data[:] = 1.0
C = (X.T @ X).tocsr(); C.setdiag(0); C.eliminate_zeros()
deg = np.asarray(X.sum(0)).ravel() + 1e-9
te['ctr'] = te.itemID.map(ctr).fillna(p0).to_numpy()
te['hb'] = hour(te.timestamp) // 4
te['ctr_h'] = pd.Series(ctr_h).reindex(pd.MultiIndex.from_arrays([te.itemID, te.hb])).to_numpy()
te['ctr_h'] = np.where(np.isnan(te.ctr_h), te.ctr, te.ctr_h)
lt = te.user_id.map(last)
te['trans1'] = trans.reindex(pd.MultiIndex.from_arrays([lt.fillna('').to_numpy(), te.itemID.to_numpy()])).fillna(0).to_numpy()
l3 = te.user_id.map(last3)
te['trans3'] = [sum(trans.get((a, b), 0) * w for a, w in zip(reversed(L), (1, .5, .25))) if isinstance(L, list) else 0
                for L, b in zip(l3, te.itemID)]
ui = users.get_indexer(te.user_id); ii = items.get_indexer(te.itemID); ok = (ui >= 0) & (ii >= 0)
cf = np.zeros(len(te))
if ok.any():
    XU = X[ui[ok]]
    Cn = C.multiply(1 / np.sqrt(np.outer(np.ones(1), deg))).tocsc()
    cf[ok] = np.asarray(XU.multiply(Cn[:, ii[ok]].T).sum(1)).ravel()
te['cf'] = cf
def hit(col, tieb='ctr'):
    s = te[col].to_numpy() + 1e-6 * te[tieb].rank(pct=True).to_numpy()
    d = pd.DataFrame({'e': te.event_id.values, 's': s, 'y': te.clicked.values})
    d['r'] = d.groupby('e').s.rank(ascending=False, method='max')
    return (d[d.y == 1].r == 1).mean()
print(f"test clicked slates {te.event_id.nunique():,} | random hit@1 = 0.333")
for col, desc in [('ctr', 'global item CTR'), ('ctr_h', 'item CTR by time of day (4h buckets)'),
                  ('trans1', 'transition from last click'), ('trans3', 'transition from last 3 clicks'),
                  ('cf', 'co-click similarity to user history')]:
    print(f"  {desc:<42} hit@1 {hit(col):.4f}   coverage {(te[col] > 0).mean():.1%}")

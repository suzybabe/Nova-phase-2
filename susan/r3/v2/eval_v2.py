"""Validate v2 settings on the same split as r3_eval (train 6-7 Jan, test 8 Jan clicked slates). CPU, ~1-2 min.
Run ON THE VM:  /opt/techarena/venvs/torch/bin/python eval_v2.py ~/data/round-3/phase-2-round-3-input
"""
import sys
import numpy as np, pandas as pd
import solution as S

fl, cat, _ = S.load(sys.argv[1])
cut = int(fl.timestamp.max()) // 86400 * 86400
tr, te = fl[fl.timestamp < cut], fl[fl.timestamp >= cut]
ec = te.groupby("event_id").clicked.sum(); te = te[te.event_id.isin(ec[ec == 1].index)].sort_values(["event_id", "itemID"])
g = te.groupby("event_id", sort=False); first = g.head(1)
K = int(g.size().max()); e = pd.factorize(te.event_id)[0]; pos = g.cumcount().to_numpy()
ITEMS = np.full((len(first), K), "", dtype=object); LAB = np.zeros((len(first), K)); MSK = np.zeros((len(first), K), bool)
ITEMS[e, pos] = te.itemID.to_numpy(); LAB[e, pos] = te.clicked.to_numpy(); MSK[e, pos] = True
users = first.user_id.to_numpy()
print(f"test clicked slates {len(first):,} | starter 0.3955 | v1 0.3885 | random 0.333 | eval#1 best 0.4230")

def hit(sc):
    sc = np.where(MSK, sc, -np.inf); c = np.where(LAB == 1, sc, 0.0).sum(1)
    r = (sc >= c[:, None]).sum(1); return (r == 1).mean(), (1 / r).mean()

for prior, hl, beta in [(200, 1.0, 0.0), (200, 1.0, 0.2), (200, 1.0, 0.5), (200, 1.0, 1.0), (200, 1.0, 2.0), (200, 1.0, 3.0),
                        (200, None, 0.5), (200, None, 1.0)]:
    m = S.ItemQuality(prior=prior, half_life_d=hl, beta=beta).fit(tr, cat)
    h, mrr = hit(m.score_pairs(users, ITEMS))
    print(f"prior={prior} half_life={str(hl):<5} beta={beta:<4} hit@1 {h:.4f} MRR {mrr:.4f}", flush=True)

"""Round 3 validation: train on 6-7 Jan, test on 8 Jan's real (random, 3-item) slates.
Compares the organisers' starter settings vs our v1 with the SAME code path.
Metrics on clicked test slates (ties count against us):
  slate hit@1 / MRR : rank the shown items; is the clicked one on top?   (fair: slates are random)
  catalog MRR / top-10 : rank of the clicked item among ALL items       (how good our single pick is)
Run ON THE VM:  /opt/techarena/venvs/torch/bin/python r3_eval.py ~/data/round-3/phase-2-round-3-input [n_test]
"""
import sys, time
import numpy as np, pandas as pd
import solution as S

data_dir = sys.argv[1]
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
fl, cat, _ = S.load(data_dir)
cut = int(fl.timestamp.max()) // 86400 * 86400            # start of the last day (8 Jan)
tr, te = fl[fl.timestamp < cut], fl[fl.timestamp >= cut]
ec = te.groupby("event_id").clicked.sum()
te = te[te.event_id.isin(ec[ec == 1].index)]
ev = te.event_id.drop_duplicates().sample(min(N_TEST, te.event_id.nunique()), random_state=0)
te = te[te.event_id.isin(ev)].sort_values(["event_id", "itemID"])
g = te.groupby("event_id", sort=False)
first = g.head(1)
users, ts = first.user_id.to_numpy(), first.timestamp.to_numpy()
K = int(g.size().max())
ITEMS = np.full((len(first), K), "", dtype=object); LAB = np.zeros((len(first), K)); MSK = np.zeros((len(first), K), bool)
pos = g.cumcount().to_numpy(); e = pd.factorize(te.event_id)[0]
ITEMS[e, pos] = te.itemID.to_numpy(); LAB[e, pos] = te.clicked.to_numpy(); MSK[e, pos] = True
clicked_item = te[te.clicked == 1].set_index("event_id").itemID.reindex(first.event_id).to_numpy()
warm = np.isin(users, tr[tr.clicked == 1].user_id.unique())
allcat = pd.DataFrame({"itemID": sorted(set(fl.itemID) | set(cat.itemID)), "valid_from": 0, "valid_to": 2**62})
print(f"train events {tr.event_id.nunique():,} (before {pd.to_datetime(cut, unit='s')}) | test clicked slates {len(first):,} "
      f"| warm users {warm.mean():.1%} | candidate items {len(allcat):,}\n", flush=True)

for name, cfg in [("starter", S.STARTER_CFG), ("v1", S.CFG)]:
    t0 = time.time()
    m = S.NovaSeq(cfg).fit(tr, cat)
    fit_s = time.time() - t0
    sc = m.score_items(users, ts, ITEMS)
    sc = np.where(MSK, sc, -np.inf)
    sc_click = np.where(LAB == 1, sc, 0.0).sum(1)
    rank = (sc >= sc_click[:, None]).sum(1)          # ties count against
    hit, mrr = (rank == 1).mean(), (1 / rank).mean()
    iv_cat = allcat.itemID.to_numpy(); colidx = pd.Index(iv_cat).get_indexer(clicked_item)
    crank = np.zeros(len(users))
    for b, cs in m.catalog_scores(users, ts, allcat):
        cc = cs[np.arange(len(b)), colidx[b]]
        crank[b] = (cs >= cc[:, None]).sum(1)
    print(f"{name:<8} slate hit@1 {hit:.4f}  MRR {mrr:.4f} | catalog MRR {np.mean(1/crank):.4f}  top10 {np.mean(crank<=10):.4f} "
          f"| warm hit@1 {(rank[warm]==1).mean():.4f} cold hit@1 {(rank[~warm]==1).mean():.4f} | fit {fit_s:.0f}s on {m.dev}", flush=True)

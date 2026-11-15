import sys, importlib.util, numpy as np, pandas as pd
R3 = "/home/user/data/round-3"; DATA = R3 + "/phase-2-round-3-input"
def mod(path, name):
    sys.path.insert(0, path.rsplit("/", 1)[0])
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    sys.path.pop(0); return m
X = pd.read_csv(DATA + "/x_pred.csv", dtype={"rowid": str, "user_id": str}).drop_duplicates("rowid")
L = pd.read_csv(R3 + "/phase-2-round-3-labels.csv", dtype={"event_id": str, "user_id": str, "itemID": str})
H = L[L.timestamp < int(X.timestamp.min())]
prev = set(zip(H[H.clicked == 1].user_id, H[H.clicked == 1].itemID))
L = L[L.timestamp >= int(X.timestamp.min())]
ev = L.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]].merge(X[["rowid", "user_id", "timestamp"]], on=["user_id", "timestamp"])
L = L.merge(ev[["event_id", "rowid"]], on="event_id")
slate = L.groupby("rowid").itemID.apply(set)
clk = L[L.clicked == 1].drop_duplicates("rowid").set_index("rowid").itemID
uid = X.set_index("rowid").user_id
rc = [(r, [i for i in slate[r] if (uid[r], i) in prev]) for r in clk.index]
withrc = [(r, its) for r, its in rc if its]
won = np.mean([clk[r] in its for r, its in withrc]) if withrc else float("nan")
print(f"(1) clicked pooled slates containing an item this user clicked on 6-8 Jan: {len(withrc):,} of {len(clk):,} | that re-click item was chosen {won:.1%} (mean slate size there {np.mean([len(slate[r]) for r, _ in withrc]):.2f})", flush=True)
def replay(name, picks, rows=None):
    p = pd.Series(picks); idx = slate.index if rows is None else slate.index[slate.index.isin(rows)]
    inS = np.array([p.get(r) in slate[r] for r in idx]); c = clk[clk.index.isin(idx[inS])]
    win = (p.reindex(c.index).to_numpy() == c.to_numpy())
    print(f"{name:<44} rows {len(idx):>7,} | pick in slate {inS.mean():6.1%} | clicked scored {len(c):>7,} | REPLAY {100*win.mean():6.2f}", flush=True)
    return p
V4 = mod("/home/user/susan_r3/v4/solution.py", "v4s"); fl, cat, xp = V4.load(DATA); xp = xp.drop_duplicates("rowid").reset_index(drop=True)
def run(prior=200, hl=1.0, beta=0.0, g_clk=0.0):
    m = V4.ItemQuality(prior=prior, half_life_d=hl, beta=beta, g_clk=g_clk).fit(fl, cat); return dict(zip(xp.rowid, m.recommend(xp, cat)))
p4 = replay("(2) v4 submitted (beta .5, reclick -5)", run(beta=0.5, g_clk=-5.0))
isrc = np.array([(uid[r], i) in prev for r, i in p4.items()]); rcrows = set(p4.index[isrc])
print(f"    v4 rows whose pick is a re-click item: {isrc.mean():.1%}")
replay("    v4 | re-click rows only", p4, rcrows); replay("    v4 | other rows only", p4, set(p4.index) - rcrows)
replay("(3) quality-first q1 (beta 0)", run())
replay("    quality-first + re-click bonus (-5)", run(g_clk=-5.0))
replay("    quality-first + re-click bonus (-1)", run(g_clk=-1.0))
replay("    co-click only v2 (beta .5)", run(beta=0.5))
for tag, w in [("Darling R3 Nova User (reclick 1.5)", 1.5), ("Nova User appeal only (reclick 0)", 0.0)]:
    try:
        NU = mod("/home/user/chioma_eval/solution.py", "nus" + str(w))
        NU.nu.Scorer.__init__.__defaults__ = (0.0, w, 0.0, 10)
        replay(tag, dict(x for b in NU.predict(DATA) for x in b))
    except Exception as e:
        print(tag, "failed:", e)

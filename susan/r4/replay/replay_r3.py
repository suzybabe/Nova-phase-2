import sys, importlib.util, numpy as np, pandas as pd
R3 = "/home/user/data/round-3"; DATA = R3 + "/phase-2-round-3-input"
def mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
X = pd.read_csv(DATA + "/x_pred.csv", dtype={"rowid": str, "user_id": str}).drop_duplicates("rowid")
L = pd.read_csv(R3 + "/phase-2-round-3-labels.csv", dtype={"event_id": str, "user_id": str, "itemID": str})
L = L[L.timestamp >= int(X.timestamp.min())]
ev = L.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]].merge(X[["rowid", "user_id", "timestamp"]], on=["user_id", "timestamp"])
L = L.merge(ev[["event_id", "rowid"]], on="event_id")
slate = L.groupby("rowid").itemID.apply(set)
clk = L[L.clicked == 1].drop_duplicates("rowid").set_index("rowid").itemID
print(f"pooled slates matched to x_pred rows: {len(slate):,} | with a click: {len(clk):,} | mean slate size {slate.map(len).mean():.2f}")
def replay(name, picks):
    p = pd.Series(picks)
    inS = np.array([p.get(r) in s for r, s in slate.items()])
    c = clk[clk.index.isin(slate.index[inS])]
    win = (p.reindex(c.index).to_numpy() == c.to_numpy())
    print(f"{name:<34} our pick in slate {inS.mean():6.1%} | clicked slates scored {len(c):>7,} | REPLAY SCORE {100*win.mean():6.2f} | distinct picks {p.nunique():,}", flush=True)
def run_pred(path, name):
    m = mod(path, name); return dict(x for b in m.predict(DATA) for x in b)
replay("v4 (submitted R3, official 42.65)", run_pred("/home/user/susan_r3/v4/solution.py", "v4s"))
replay("v2 (personal co-click)", run_pred("/home/user/susan_r3/v2/solution.py", "v2s"))
Q = mod("/home/user/susan_r4/q1/solution.py", "q1s"); fl, cat, xp = Q.load(DATA); xp = xp.drop_duplicates("rowid").reset_index(drop=True)
for prior in [20, 50, 200, 1000, 5000]:
    for hl in [0.5, 1.0, 2.0, None]:
        m = Q.ItemQuality(prior=prior, half_life_d=hl, beta=0.0).fit(fl, cat)
        replay(f"quality-first prior={prior} hl={hl}", dict(zip(xp.rowid, m.recommend(xp, cat))))

import sys, importlib.util, numpy as np, pandas as pd
R3 = "/home/user/data/round-3"; DATA = R3 + "/phase-2-round-3-input"; NUDIR = "/home/user/susan_r4/nu"
sys.path.insert(0, NUDIR)
spec = importlib.util.spec_from_file_location("nusol", NUDIR + "/solution.py"); NU = importlib.util.module_from_spec(spec); spec.loader.exec_module(NU)
X = pd.read_csv(DATA + "/x_pred.csv", dtype={"rowid": str, "user_id": str}).drop_duplicates("rowid")
L = pd.read_csv(R3 + "/phase-2-round-3-labels.csv", dtype={"event_id": str, "user_id": str, "itemID": str})
L = L[L.timestamp >= int(X.timestamp.min())]
ev = L.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]].merge(X[["rowid", "user_id", "timestamp"]], on=["user_id", "timestamp"])
L = L.merge(ev[["event_id", "rowid"]], on="event_id")
slate = L.groupby("rowid").itemID.apply(set); clk = L[L.clicked == 1].drop_duplicates("rowid").set_index("rowid").itemID
def replay(name, p):
    p = pd.Series(p); inS = np.array([p.get(r) in slate[r] for r in slate.index]); c = clk[clk.index.isin(slate.index[inS])]
    print(f"{name:<36} pick in slate {inS.mean():6.1%} | clicked scored {len(c):>7,} | REPLAY {100*(p.reindex(c.index).to_numpy()==c.to_numpy()).mean():6.2f}", flush=True)
for oe in [20.0, 5.0, 80.0]:
    for w in ([0.5, 1.0, 1.5, 2.0, 3.0, 5.0] if oe == 20.0 else [1.5]):
        NU.nu.OE_S = oe; NU.nu.Scorer.__init__.__defaults__ = (0.0, w, 0.0, 10)
        replay(f"Nova User OE_S={oe:<5} W_RECLICK={w}", dict(x for b in NU.predict(DATA) for x in b))

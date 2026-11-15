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
def picks(oe):
    NU.nu.OE_S = oe; NU.nu.Scorer.__init__.__defaults__ = (0.0, 1.5, 0.0, 10)
    return pd.Series(dict(x for b in NU.predict(DATA) for x in b))
base = picks(20.0); S = slate.reindex(clk.index)
for oe in [5.0, 10.0, 40.0, 80.0]:
    alt = picks(oe); a, b = base.reindex(clk.index), alt.reindex(clk.index)
    both = np.array([x in s and y in s for x, y, s in zip(a, b, S)])
    diff = both & (a.to_numpy() != b.to_numpy())
    wa = (a.to_numpy()[diff] == clk.to_numpy()[diff]).sum(); wb = (b.to_numpy()[diff] == clk.to_numpy()[diff]).sum()
    print(f"OE_S 20 vs {oe:<5}: rows both in slate {both.sum():>7,} | picks differ {diff.sum():>6,} | user chose OE20 pick {wa:>5,} vs OE{oe:g} pick {wb:>5,} | net for OE{oe:g}: {wb-wa:+,} (= {100*(wb-wa)/len(clk):+.2f} replay points over all clicked slates)", flush=True)
